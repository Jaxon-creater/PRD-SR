import os, argparse
# from yaml import safe_load as yaml_load
from json import dumps as json_dumps

def parse_args():
    parser = argparse.ArgumentParser(description='SDR Arguments')
    parser.add_argument('--desc', type=str, default='')

   #Configuration Arguments
    parser.add_argument('--cuda',   type=str, default='0')
    parser.add_argument('--seed',   type=int, default=2023)

   #Model Arguments
    parser.add_argument('--n_hid',     type=int,   default=64)
    parser.add_argument('--n_layers',  type=int,   default=2)
    parser.add_argument('--s_layers', type=int, default=2)
    parser.add_argument('--weight', type=bool, default=True, help='Add linear weight or not')



    #Train Arguments
    parser.add_argument('--dropout', type=float, default=0)

    #Optimization Arguments
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--difflr', type=float, default=1e-3)
    parser.add_argument('--reg',        type=float, default=5e-3)
    parser.add_argument('--decay',      type=float, default=0.985)
    parser.add_argument('--decay_step', type=int,   default=1)
    parser.add_argument('--n_epoch',    type=int,   default=175)
    parser.add_argument('--batch_size', type=int,   default=2048)
    parser.add_argument('--patience',   type=int,   default=20)

    # Valid/Test Arguments 
    parser.add_argument('--topk',            type=int, default=20)
    parser.add_argument('--test_batch_size', type=int, default=1024)

    #Data Arguments
    parser.add_argument('--dataset',       type=str, default="amazon-beauty")
    parser.add_argument('--num_workers',   type=int, default=0)
    parser.add_argument('--save_name', type=str, default='tem')
    parser.add_argument('--checkpoint',    type=str, default="./Model/amazon-beauty/_tem_.pth")
    parser.add_argument('--model_dir',     type=str, default="./Model/amazon-beauty/")
   
    # params for the denoiser
    parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
    parser.add_argument('--dims', type=int, default=64, help='the dims for the DNN')
    parser.add_argument('--norm', type=bool, default=True, help='Normalize the input or not')
    parser.add_argument('--emb_size', type=int, default=8, help='timestep embedding size')

    # params for diffusions
    parser.add_argument('--steps', type=int, default=20, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=0.01, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.01, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
    parser.add_argument('--reweight', type=bool, default=True,
                        help='assign different weight to different timestep or not')

    # params for personalized noise scheduling
    parser.add_argument('--activity_weight', type=float, default=0.8,
                        help='weight for user activity in personalized noise scheduling')
    parser.add_argument('--influence_weight', type=float, default=0.2,
                        help='weight for user influence in personalized noise scheduling')
    parser.add_argument('--min_noise_scale', type=float, default=0.1,
                        help='minimum noise scale for high-activity/high-influence users')
    parser.add_argument('--max_noise_scale', type=float, default=1.0,
                        help='maximum noise scale for low-activity/low-influence users')

    # params for semantic features
    parser.add_argument('--semantic_dim', type=int, default=384,
                        help='dimension of semantic features from all-MiniLM-L6-v2')
    parser.add_argument('--review_text_path', type=str, default='datasets/amazon/review_texts.csv',
                        help='path to review texts CSV file')
    parser.add_argument('--use_semantic', type=bool, default=True,
                        help='whether to use semantic features')
    parser.add_argument('--semantic_cache_dir', type=str, default='./cache/semantic_features/',
                        help='directory to cache computed semantic features')
    parser.add_argument('--semantic_fusion_mode', type=str, default='residual',
                        choices=['residual', 'multimodal'],
                        help='how to fuse semantic features: residual preserves RDGC as the main representation')
    parser.add_argument('--user_semantic_residual_weight', type=float, default=0.5,
                        help='residual weight for adding projected user text features to RDGC user embeddings')
    parser.add_argument('--item_semantic_residual_weight', type=float, default=0.5,
                        help='residual weight for adding projected item text features to item embeddings')

    # params for dual-graph collaboration
    parser.add_argument('--use_dual_graph', type=bool, default=True,
                        help='whether to use dual-graph collaboration mechanism')
    parser.add_argument('--dual_graph_temp', type=float, default=0.1,
                        help='temperature for dual-graph attention')
    parser.add_argument('--kl_weight', type=float, default=0.03,
                        help='weight for KL divergence alignment loss')
    parser.add_argument('--rdgc_social_shift_penalty', type=float, default=1.0,
                        help='penalty strength for diffusion-induced social shifts in RDGC')
    parser.add_argument('--rdgc_temp', type=float, default=1.0,
                        help='softmax temperature for RDGC reliability gates')
    parser.add_argument('--rdgc_residual_weight', type=float, default=0.2,
                        help='weight for preserving original graph representations in RDGC')
    parser.add_argument('--rdgc_cross_weight', type=float, default=0.3,
                        help='strength of cross-view representation in RDGC')
    parser.add_argument('--rdgc_use_cross_view', type=bool, default=True,
                        help='whether to use cross-view representation in RDGC')

    # params for multimodal fusion
    parser.add_argument('--use_multimodal', type=bool, default=True,
                        help='whether to use multimodal feature fusion')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='number of attention heads in multimodal fusion')
    parser.add_argument('--fusion_dropout', type=float, default=0.1,
                        help='dropout rate in multimodal fusion')

    # params for contrastive learning
    parser.add_argument('--use_contrastive', type=bool, default=True,
                        help='whether to use contrastive learning')
    parser.add_argument('--cl_temperature', type=float, default=0.2,
                        help='temperature for contrastive learning')
    parser.add_argument('--diffusion_cl_weight', type=float, default=0.001,
                        help='weight for diffusion-aware contrastive loss')
    parser.add_argument('--cross_modal_cl_weight', type=float, default=0.001,
                        help='weight for cross-modal contrastive loss')

    return parser.parse_args()

args = parse_args()


# ciao 5e-3, 1e-3, 1 1e-2, 20, emb_size:16



