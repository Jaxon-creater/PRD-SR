import torch, pickle, time, os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from utils.param import args
from datasets.DataHander import DataHandler
from models.model import SDNet, GCNModel
from models.PersonalizedNoiseScheduler import PersonalizedNoiseScheduler
from models.SemanticFeatureGenerator import SemanticFeatureGenerator, SemanticProjection
from models.MultimodalFusion import DualGraphCollaboration, MultimodalFeatureFusion, ContrastiveLearning

from utils.utils import load_model, save_model, fix_random_seed_as
from tqdm import tqdm


from models import diffusion_process as dp
from utils.legacy_utils import *
import logging
import sys


class Coach:
    def __init__(self, handler):
        self.args = args
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        self.handler = handler
        self.train_loader = self.handler.trainloader
        self.valloader = self.handler.valloader
        self.testloader = self.handler.testloader
        self.n_user, self.n_item = self.handler.n_user, self.handler.n_item
        self.uiGraph = self.handler.ui_graph.to(self.device)
        self.uuGraph = self.handler.uu_graph.to(self.device)

        # ========== Initialize Personalized Noise Scheduler ==========
        print("[Coach] Initializing Personalized Noise Scheduler...")
        self.personalized_scheduler = PersonalizedNoiseScheduler(
            n_users=self.n_user,
            train_mat=self.handler.dataset['train'],
            trust_mat=self.handler.dataset['trust'],
            activity_weight=args.activity_weight if hasattr(args, 'activity_weight') else 0.5,
            influence_weight=args.influence_weight if hasattr(args, 'influence_weight') else 0.5,
            degree_weight=getattr(args, 'degree_weight', 0.6),
            pagerank_weight=getattr(args, 'pagerank_weight', 0.4),
            min_noise_scale=args.min_noise_scale if hasattr(args, 'min_noise_scale') else 0.3,
            max_noise_scale=args.max_noise_scale if hasattr(args, 'max_noise_scale') else 1.0
        )

        stats = self.personalized_scheduler.get_statistics()
        print(f"[Coach] Noise Scheduler Stats:")
        print(f"  Activity - mean: {stats['activity']['mean']:.4f}, std: {stats['activity']['std']:.4f}")
        print(f"  Influence - mean: {stats['influence']['mean']:.4f}, std: {stats['influence']['std']:.4f}")
        print(f"  Noise Scale - mean: {stats['noise_scale']['mean']:.4f}, std: {stats['noise_scale']['std']:.4f}")

        # ========== Initialize Semantic Feature Generator ==========
        self.use_semantic = getattr(args, 'use_semantic', True)
        self.user_semantic_features = None
        self.item_semantic_features = None

        if self.use_semantic:
            print("[Coach] Initializing Semantic Feature Generator...")
            self._initialize_semantic_features()

        # ========== Initialize Base Models ==========
        self.GCNModel = GCNModel(args, self.n_user, self.n_item).to(self.device)

        output_dims = [args.dims] + [args.n_hid]
        input_dims = output_dims[::-1]
        self.SDNet = SDNet(input_dims, output_dims, args.emb_size, time_type="cat", norm=args.norm).to(self.device)

        self.DiffProcess = dp.DiffusionProcess(
            args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max,
            args.steps, self.device, personalized_scheduler=self.personalized_scheduler
        ).to(self.device)

        # ========== Initialize Semantic Projection ==========
        self.semantic_fusion_mode = getattr(args, 'semantic_fusion_mode', 'residual')
        if self.use_semantic:
            print("[Coach] Initializing User/Item Semantic Projection Layers...")
            self.user_semantic_projection = SemanticProjection(
                input_dim=getattr(args, 'semantic_dim', 384),
                output_dim=args.n_hid,
                dropout=getattr(args, 'fusion_dropout', 0.1)
            ).to(self.device)
            self.item_semantic_projection = SemanticProjection(
                input_dim=getattr(args, 'semantic_dim', 384),
                output_dim=args.n_hid,
                dropout=getattr(args, 'fusion_dropout', 0.1)
            ).to(self.device)
        else:
            self.user_semantic_projection = None
            self.item_semantic_projection = None

        # ========== Initialize Dual-Graph Collaboration ==========
        self.use_dual_graph = getattr(args, 'use_dual_graph', True)
        if self.use_dual_graph:
            print("[Coach] Initializing Dual-Graph Collaboration Module...")
            self.dual_graph_collab = DualGraphCollaboration(
                n_users=self.n_user,
                n_items=self.n_item,
                hidden_dim=args.n_hid,
                temperature=getattr(args, 'dual_graph_temp', 0.1),
                interaction_reliability=self.personalized_scheduler.user_activity,
                social_struct_reliability=self.personalized_scheduler.user_influence,
                social_shift_penalty=getattr(args, 'rdgc_social_shift_penalty', 1.0),
                use_cross_view=getattr(args, 'rdgc_use_cross_view', True),
                reliability_temperature=getattr(args, 'rdgc_temp', 1.0),
                confidence_threshold=getattr(args, 'rdgc_confidence_threshold', 0.5),
                fallback_slope=getattr(args, 'rdgc_fallback_slope', 10.0)
            ).to(self.device)
        else:
            self.dual_graph_collab = None

        # ========== Initialize Multimodal Feature Fusion ==========
        self.use_multimodal = getattr(args, 'use_multimodal', True)
        if self.use_multimodal and self.use_semantic and self.semantic_fusion_mode == 'multimodal':
            print("[Coach] Initializing Multimodal Feature Fusion Module...")
            self.multimodal_fusion = MultimodalFeatureFusion(
                hidden_dim=args.n_hid,
                n_heads=getattr(args, 'n_heads', 4),
                dropout=getattr(args, 'fusion_dropout', 0.1)
            ).to(self.device)
        else:
            self.multimodal_fusion = None

        # ========== Initialize Contrastive Learning ==========
        self.use_contrastive = getattr(args, 'use_contrastive', True)
        if self.use_contrastive:
            print("[Coach] Initializing Contrastive Learning Module...")
            self.contrastive_learning = ContrastiveLearning(
                hidden_dim=args.n_hid,
                temperature=getattr(args, 'cl_temperature', 0.07)
            ).to(self.device)
        else:
            self.contrastive_learning = None

        # ========== Setup Optimizers ==========
        self._setup_optimizers()

        self.train_loss = []
        self.his_recall = []
        self.his_ndcg = []

        print("[Coach] Initialization complete!")
        print(f"  - Use Semantic Features: {self.use_semantic}")
        print(f"  - Semantic Fusion Mode: {self.semantic_fusion_mode}")
        print(f"  - Use Dual-Graph Collaboration: {self.use_dual_graph}")
        print(f"  - Use Multimodal Fusion: {self.multimodal_fusion is not None}")
        print(f"  - Use Contrastive Learning: {self.use_contrastive}")

    def _initialize_semantic_features(self):
        """Initialize or load semantic features from review texts."""
        semantic_cache_dir = getattr(args, 'semantic_cache_dir', './cache/semantic_features/')
        cache_path_user = os.path.join(semantic_cache_dir, f'{args.dataset}_user_semantic.pt')
        cache_path_item = os.path.join(semantic_cache_dir, f'{args.dataset}_item_semantic.pt')

        # Try to load from cache
        if os.path.exists(cache_path_user) and os.path.exists(cache_path_item):
            print("[Coach] Loading cached semantic features...")
            self.user_semantic_features = torch.load(cache_path_user).to(self.device)
            self.item_semantic_features = torch.load(cache_path_item).to(self.device)
            print(f"[Coach] Loaded user semantic features: {self.user_semantic_features.shape}")
            print(f"[Coach] Loaded item semantic features: {self.item_semantic_features.shape}")
        else:
            # Generate semantic features from review texts
            review_text_path = getattr(args, 'review_text_path', 'datasets/yelp/review_texts.csv')

            if os.path.exists(review_text_path):
                print(f"[Coach] Generating semantic features from {review_text_path}...")
                try:
                    semantic_generator = SemanticFeatureGenerator(
                        model_path=getattr(
                            args, 'minilm_model_path', 'models/all-MiniLM-L6-v2'
                        ),
                        device=self.device
                    )
                    self.user_semantic_features, self.item_semantic_features = \
                        semantic_generator.load_and_process_reviews(
                            review_text_path, self.n_user, self.n_item
                        )

                    # Move to device
                    self.user_semantic_features = self.user_semantic_features.to(self.device)
                    self.item_semantic_features = self.item_semantic_features.to(self.device)

                    # Save to cache
                    os.makedirs(semantic_cache_dir, exist_ok=True)
                    torch.save(self.user_semantic_features.cpu(), cache_path_user)
                    torch.save(self.item_semantic_features.cpu(), cache_path_item)
                    print(f"[Coach] Cached semantic features to {semantic_cache_dir}")
                except Exception as e:
                    raise RuntimeError(
                        "Failed to generate semantic features from the local MiniLM model. "
                        "Verify --minilm_model_path and the review CSV."
                    ) from e
            else:
                raise FileNotFoundError(f"Review text file not found: {review_text_path}")

    def _setup_optimizers(self):
        """Setup optimizers for all trainable modules."""
        # Collect all parameters
        gcn_params = list(self.GCNModel.parameters())
        diff_params = list(self.SDNet.parameters())

        additional_params = []
        if self.user_semantic_projection is not None:
            additional_params.extend(list(self.user_semantic_projection.parameters()))
        if self.item_semantic_projection is not None:
            additional_params.extend(list(self.item_semantic_projection.parameters()))
        if self.dual_graph_collab is not None:
            additional_params.extend(list(self.dual_graph_collab.parameters()))
        if self.multimodal_fusion is not None:
            additional_params.extend(list(self.multimodal_fusion.parameters()))
        if self.contrastive_learning is not None:
            additional_params.extend(list(self.contrastive_learning.parameters()))

        # Optimizer 1: GCN + additional modules
        self.optimizer1 = torch.optim.Adam([
            {'params': gcn_params, 'weight_decay': 0},
            {'params': additional_params, 'weight_decay': 0, 'lr': args.lr * 0.5}
        ], lr=args.lr)

        # Optimizer 2: Diffusion network
        self.optimizer2 = torch.optim.Adam([
            {'params': diff_params, 'weight_decay': 0},
        ], lr=args.difflr)

        self.scheduler1 = torch.optim.lr_scheduler.StepLR(
            self.optimizer1, step_size=args.decay_step, gamma=args.decay
        )
        self.scheduler2 = torch.optim.lr_scheduler.StepLR(
            self.optimizer2, step_size=args.decay_step, gamma=args.decay
        )

    def _semantic_enabled(self):
        return (
            self.use_semantic
            and self.user_semantic_projection is not None
            and self.item_semantic_projection is not None
            and self.user_semantic_features is not None
            and self.item_semantic_features is not None
        )

    def _apply_user_semantic_residual(self, user_emb, user_idx):
        if not self._semantic_enabled():
            return user_emb, None

        user_semantic = self.user_semantic_features[user_idx]
        user_semantic_proj = self.user_semantic_projection(user_semantic)
        weight = getattr(args, 'user_semantic_residual_weight', 0.05)
        return user_emb + weight * user_semantic_proj, user_semantic_proj

    def _apply_item_semantic_residual(self, item_emb, item_idx=None):
        if not self._semantic_enabled():
            return item_emb, None

        if item_idx is None:
            item_semantic = self.item_semantic_features
        else:
            item_semantic = self.item_semantic_features[item_idx]

        item_semantic_proj = self.item_semantic_projection(item_semantic)
        weight = getattr(args, 'item_semantic_residual_weight', 0.05)
        return item_emb + weight * item_semantic_proj, item_semantic_proj

    def train(self):
        args = self.args
        self.save_history = True
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        log_save = './History/' + args.dataset + '/'
        os.makedirs(log_save, exist_ok=True)
        log_file = args.save_name
        fname = f'{log_file}.txt'
        fh = logging.FileHandler(os.path.join(log_save, fname))
        fh.setFormatter(logging.Formatter(log_format))
        logger = logging.getLogger()
        logger.addHandler(fh)
        logger.info(args)
        logger.info('================')
        best_recall, best_ndcg, best_epoch, wait = 0, 0, 0, 0
        start_time = time.time()

        for self.epoch in range(1, args.n_epoch + 1):
            epoch_losses = self.train_one_epoch()
            self.train_loss.append(epoch_losses)
            print('epoch {} done! elapsed {:.2f}s, epoch_losses {}'.format(
                self.epoch, time.time() - start_time, epoch_losses
            ), flush=True)

            if self.epoch % 5 == 0:
                recall, ndcg = self.test(self.testloader)

                self.his_recall.append(recall)
                self.his_ndcg.append(ndcg)
                cur_best = recall + ndcg > best_recall + best_ndcg

                if cur_best:
                    best_recall, best_ndcg, best_epoch = recall, ndcg, self.epoch
                    wait = 0
                else:
                    wait += 1

                logger.info('+ epoch {} tested, elapsed {:.2f}s, Recall@{}: {:.4f}, NDCG@{}: {:.4f}'.format(
                    self.epoch, time.time() - start_time, args.topk, recall, args.topk, ndcg))

                if args.model_dir and cur_best:
                    os.makedirs(args.model_dir, exist_ok=True)
                    desc = args.save_name
                    perf = ''
                    fname = f'{args.desc}_{desc}_{perf}.pth'
                    save_model(self.GCNModel, self.SDNet, os.path.join(args.model_dir, fname),
                               self.optimizer1, self.optimizer2)

            if self.save_history:
                self.saveHistory()

            if wait >= args.patience:
                print(f'Early stop at epoch {self.epoch}, best epoch {best_epoch}')
                break

        print(f'Best Recall@{args.topk} {best_recall:.6f}, NDCG@{args.topk} {best_ndcg:.6f},', flush=True)

    def train_one_epoch(self):
        """Train one epoch with all modules integrated."""
        # Set all modules to training mode
        self.SDNet.train()
        self.GCNModel.train()
        if self.user_semantic_projection is not None:
            self.user_semantic_projection.train()
        if self.item_semantic_projection is not None:
            self.item_semantic_projection.train()
        if self.dual_graph_collab is not None:
            self.dual_graph_collab.train()
        if self.multimodal_fusion is not None:
            self.multimodal_fusion.train()
        if self.contrastive_learning is not None:
            self.contrastive_learning.train()

        dataloader = self.train_loader
        # Extended loss tracking: [bpr, reg, diff, kl, cl_diff, cl_cross]
        epoch_losses = [0] * 6
        dataloader.dataset.negSampling()
        tqdm_dataloader = tqdm(dataloader)
        since = time.time()

        for iteration, batch in enumerate(tqdm_dataloader):
            user_idx, pos_idx, neg_idx = batch
            user_idx = user_idx.long().cuda()
            pos_idx = pos_idx.long().cuda()
            neg_idx = neg_idx.long().cuda()

            # ========== Step 1: GCN Forward Pass ==========
            uiEmbeds, uuEmbeds = self.GCNModel(self.uiGraph, self.uuGraph, True)
            uEmbeds = uiEmbeds[:self.n_user]
            iEmbeds = uiEmbeds[self.n_user:]

            user_interaction = uEmbeds[user_idx]
            user_social = uuEmbeds[user_idx]
            pos = iEmbeds[pos_idx]
            neg = iEmbeds[neg_idx]

            # ========== Step 2: Diffusion Process ==========
            # Store noisy embeddings for contrastive learning
            noisy_social = user_social.clone()

            uu_terms = self.DiffProcess.caculate_losses(
                self.SDNet, user_social, args.reweight, user_indices=user_idx
            )
            uuelbo = uu_terms["loss"].mean()
            denoised_social = uu_terms["pred_xstart"]  # Clean social embeddings

            diffloss = uuelbo

            # ========== Step 3: Dual-Graph Collaboration ==========
            kl_loss = torch.tensor(0.0, device=self.device)
            if self.use_dual_graph and self.dual_graph_collab is not None:
                user_rdgc, rdgc_social, kl_loss = self.dual_graph_collab(
                    user_social,
                    user_interaction,
                    self.GCNModel.embedding_dict['user_emb'][user_idx],
                    denoised_social_emb=denoised_social,
                    user_indices=user_idx
                )
            else:
                user_rdgc = user_interaction + denoised_social
                rdgc_social = denoised_social

            # ========== Step 4: Semantic Feature Processing and Fusion ==========
            user_semantic_proj = None
            if self.semantic_fusion_mode == 'multimodal' and self.multimodal_fusion is not None and self._semantic_enabled():
                user_semantic = self.user_semantic_features[user_idx]
                user_semantic_proj = self.user_semantic_projection(user_semantic)
                user_fused, modality_weights = self.multimodal_fusion(
                    semantic_feat=user_semantic_proj,
                    social_feat=rdgc_social,
                    interaction_feat=user_interaction,
                    base_feat=user_interaction
                )
            else:
                user_fused, user_semantic_proj = self._apply_user_semantic_residual(
                    user_rdgc, user_idx
                )
                pos, _ = self._apply_item_semantic_residual(pos, pos_idx)
                neg, _ = self._apply_item_semantic_residual(neg, neg_idx)

            # ========== Step 5: Contrastive Learning ==========
            cl_diff_loss = torch.tensor(0.0, device=self.device)
            cl_cross_loss = torch.tensor(0.0, device=self.device)

            if self.use_contrastive and self.contrastive_learning is not None:
                # Task A: Diffusion-Aware Contrastive Learning
                cl_diff_loss = self.contrastive_learning.diffusion_aware_contrastive_loss(
                    noisy_social, rdgc_social
                )

                # Task B: Cross-modal Contrastive Learning (if semantic features available)
                cross_cl_weight = getattr(args, 'cross_modal_cl_weight', 0.001)
                if user_semantic_proj is not None and cross_cl_weight > 0:
                    cl_cross_loss = self.contrastive_learning.cross_modal_contrastive_loss(
                        user_fused, user_semantic_proj
                    )

            # ========== Step 6: Compute Final Loss ==========
            # BPR Loss
            scoreDiff = pairPredict(user_fused, pos, neg)
            bprLoss = -(scoreDiff).sigmoid().log().sum() / args.batch_size

            # Regularization Loss
            regLoss = ((torch.norm(user_fused) ** 2 + torch.norm(pos) ** 2 + torch.norm(neg) ** 2) * args.reg) / args.batch_size

            # Total Loss
            loss = bprLoss + regLoss + diffloss

            # Add KL divergence loss
            kl_weight = getattr(args, 'kl_weight', 0.1)
            loss = loss + kl_weight * kl_loss

            # Add contrastive learning losses
            diff_cl_weight = getattr(args, 'diffusion_cl_weight', 0.1)
            cross_cl_weight = getattr(args, 'cross_modal_cl_weight', 0.001)

            loss = loss + diff_cl_weight * cl_diff_loss + cross_cl_weight * cl_cross_loss

            # Record losses
            losses = [
                bprLoss.item(),
                regLoss.item(),
                diffloss.item(),
                kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
                cl_diff_loss.item() if isinstance(cl_diff_loss, torch.Tensor) else cl_diff_loss,
                cl_cross_loss.item() if isinstance(cl_cross_loss, torch.Tensor) else cl_cross_loss
            ]

            # ========== Step 7: Backward Pass ==========
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

            epoch_losses = [x + y for x, y in zip(epoch_losses, losses)]

        if self.scheduler1 is not None:
            self.scheduler1.step()
            self.scheduler2.step()

        epoch_losses = [sum(epoch_losses)] + epoch_losses
        time_elapsed = time.time() - since
        print('Training complete in {:.4f}s'.format(time_elapsed))
        return epoch_losses

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def test(self, dataloader):
        """Test with all modules integrated."""
        # Set all modules to evaluation mode
        self.SDNet.eval()
        self.GCNModel.eval()
        if self.user_semantic_projection is not None:
            self.user_semantic_projection.eval()
        if self.item_semantic_projection is not None:
            self.item_semantic_projection.eval()
        if self.dual_graph_collab is not None:
            self.dual_graph_collab.eval()
        if self.multimodal_fusion is not None:
            self.multimodal_fusion.eval()

        Recall, NDCG = 0, 0
        num = dataloader.dataset.__len__()

        since = time.time()
        with torch.no_grad():
            uiEmbeds, uuEmbeds = self.GCNModel(self.uiGraph, self.uuGraph, True)
            uEmbeds = uiEmbeds[:self.n_user]
            iEmbeds = uiEmbeds[self.n_user:]
            all_item_fused, _ = self._apply_item_semantic_residual(iEmbeds)

            tqdm_dataloader = tqdm(dataloader)
            for iteration, batch in enumerate(tqdm_dataloader, start=1):
                user_idx, trnMask = batch
                user_idx = user_idx.long().cuda()
                trnMask = trnMask.cuda()

                user_interaction = uEmbeds[user_idx]
                user_social = uuEmbeds[user_idx]

                # Diffusion denoising
                denoised_social = self.DiffProcess.p_sample(
                    self.SDNet, user_social, args.sampling_steps,
                    args.sampling_noise, user_indices=user_idx
                )

                # Dual-graph collaboration
                if self.use_dual_graph and self.dual_graph_collab is not None:
                    user_rdgc, rdgc_social, _ = self.dual_graph_collab(
                        user_social,
                        user_interaction,
                        self.GCNModel.embedding_dict['user_emb'][user_idx],
                        denoised_social_emb=denoised_social,
                        user_indices=user_idx
                    )
                else:
                    user_rdgc = user_interaction + denoised_social
                    rdgc_social = denoised_social

                # Semantic feature processing and fusion
                if self.semantic_fusion_mode == 'multimodal' and self.multimodal_fusion is not None and self._semantic_enabled():
                    user_semantic = self.user_semantic_features[user_idx]
                    user_semantic_proj = self.user_semantic_projection(user_semantic)
                    user_fused, _ = self.multimodal_fusion(
                        semantic_feat=user_semantic_proj,
                        social_feat=rdgc_social,
                        interaction_feat=user_interaction,
                        base_feat=user_interaction
                    )
                    item_for_prediction = iEmbeds
                else:
                    user_fused, _ = self._apply_user_semantic_residual(user_rdgc, user_idx)
                    item_for_prediction = all_item_fused

                # Prediction
                allPreds = t.mm(user_fused, t.transpose(item_for_prediction, 1, 0)) * (1 - trnMask) - trnMask * 1e8
                _, topLocs = t.topk(allPreds, args.topk)
                recall, ndcg = self.calcRes(topLocs.cpu().numpy(), dataloader.dataset.tstLocs, user_idx)
                Recall += recall
                NDCG += ndcg

            time_elapsed = time.time() - since
            print('Testing complete in {:.4f}s'.format(time_elapsed))
            Recall = Recall / num
            NDCG = NDCG / num
        return Recall, NDCG

    def saveHistory(self):
        history = dict()
        history['loss'] = self.train_loss
        history['Recall'] = self.his_recall
        history['NDCG'] = self.his_ndcg
        desc = args.save_name
        perf = ''
        fname = f'{args.desc}_{desc}_{perf}.his'

        save_dir = './History/' + args.dataset + '/'
        os.makedirs(save_dir, exist_ok=True)

        with open(save_dir + fname, 'wb') as fs:
            pickle.dump(history, fs)

        # Plot total loss over epochs
        if len(self.train_loss) > 0:
            epochs = list(range(1, len(self.train_loss) + 1))
            total_losses = [loss[0] for loss in self.train_loss]

            plt.figure(figsize=(12, 5))

            # Subplot 1: Total loss
            plt.subplot(1, 2, 1)
            plt.plot(epochs, total_losses, 'b-', linewidth=2, label='Total Loss')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title(f'Training Loss - {args.dataset}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)

            # Subplot 2: Component losses
            if len(self.train_loss[0]) > 4:
                plt.subplot(1, 2, 2)
                bpr_losses = [loss[1] for loss in self.train_loss]
                diff_losses = [loss[3] for loss in self.train_loss]
                kl_losses = [loss[4] for loss in self.train_loss]
                plt.plot(epochs, bpr_losses, '-', linewidth=1.5, label='BPR Loss')
                plt.plot(epochs, diff_losses, '-', linewidth=1.5, label='Diffusion Loss')
                plt.plot(epochs, kl_losses, '-', linewidth=1.5, label='KL Loss')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.title(f'Component Losses - {args.dataset}', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=10)

            plt.tight_layout()
            img_fname = f'{args.desc}_{desc}_{perf}_loss.png'
            save_path = save_dir + img_fname
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Loss curve saved to {save_path}')
            plt.close()

        # Plot Recall and NDCG over epochs
        if len(self.his_recall) > 0 and len(self.his_ndcg) > 0:
            test_epochs = [5 * (i + 1) for i in range(len(self.his_recall))]

            plt.figure(figsize=(10, 6))
            plt.plot(test_epochs, self.his_recall, 'r-o', linewidth=2, markersize=6, label=f'Recall@{args.topk}')
            plt.plot(test_epochs, self.his_ndcg, 'g-s', linewidth=2, markersize=6, label=f'NDCG@{args.topk}')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Metric Value', fontsize=12)
            plt.title(f'Recall and NDCG over Epochs - {args.dataset}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.tight_layout()

            img_fname = f'{args.desc}_{desc}_{perf}_metrics.png'
            save_path = save_dir + img_fname
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Metrics curve saved to {save_path}')
            plt.close()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    fix_random_seed_as(args.seed)

    handler = DataHandler()
    handler.LoadData()
    app = Coach(handler)
    app.train()
