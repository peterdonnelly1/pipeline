
import argparse

from   cuda_tsne  import cuda_tsne

WHITE='\033[37;1m'
PURPLE='\033[35;1m'
DIM_WHITE='\033[37;2m'
DULL_WHITE='\033[38;2;140;140;140m'
CYAN='\033[36;1m'
MIKADO='\033[38;2;255;196;12m'
AZURE='\033[38;2;0;127;255m'
AMETHYST='\033[38;2;153;102;204m'
ASPARAGUS='\033[38;2;135;169;107m'
CHARTREUSE='\033[38;2;223;255;0m'
COQUELICOT='\033[38;2;255;56;0m'
COTTON_CANDY='\033[38;2;255;188;217m'
HOT_PINK='\033[38;2;255;105;180m'
CAMEL='\033[38;2;193;154;107m'
MAGENTA='\033[38;2;255;0;255m'
YELLOW='\033[38;2;255;255;0m'
DULL_YELLOW='\033[38;2;179;179;0m'
ARYLIDE='\033[38;2;233;214;107m'
BLEU='\033[38;2;49;140;231m'
DULL_BLUE='\033[38;2;0;102;204m'
RED='\033[38;2;255;0;0m'
PINK='\033[38;2;255;192;203m'
BITTER_SWEET='\033[38;2;254;111;94m'
PALE_RED='\033[31m'
DARK_RED='\033[38;2;120;0;0m'
ORANGE='\033[38;2;255;103;0m'
PALE_ORANGE='\033[38;2;127;63;0m'
GOLD='\033[38;2;255;215;0m'
GREEN='\033[38;2;19;136;8m'
BRIGHT_GREEN='\033[38;2;102;255;0m'
CARRIBEAN_GREEN='\033[38;2;0;204;153m'
PALE_GREEN='\033[32m'
GREY_BACKGROUND='\033[48;2;60;60;60m'

MACOSKO_COLORS = {
    "Amacrine cells": "#A5C93D",
    "Astrocytes": "#8B006B",
    "Bipolar cells": "#2000D7",
    "Cones": "#538CBA",
    "Fibroblasts": "#8B006B",
    "Horizontal cells": "#B33B19",
    "Microglia": "#8B006B",
    "Muller glia": "#8B006B",
    "Pericytes": "#8B006B",
    "Retinal ganglion cells": "#C38A1F",
    "Rods": "#538CBA",
    "Vascular endothelium": "#8B006B",
}

BOLD='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
BLINK='\033[5m'
RESET='\033[m'

CLEAR_LINE='\033[0K'
UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'
SAVE_CURSOR='\033[s'
RESTORE_CURSOR='\033[u'

FAIL    = 0
SUCCESS = 1

DEBUG   = 1

      
def main(args):

  cuda_tsne(  args, pct_test=1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--skip_tiling',                                                   type=str,   default='False'                            )                                
    p.add_argument('--skip_generation',                                               type=str,   default='False'                            )                                
    p.add_argument('--pretrain',                                                      type=str,   default='False'                            )                                
    p.add_argument('--log_dir',                                                       type=str,   default='data/dlbcl_image/logs'            )                
    p.add_argument('--base_dir',                                                      type=str,   default='/home/peter/git/pipeline'         )
    p.add_argument('--data_dir',                                                      type=str,   default='/home/peter/git/pipeline/dataset' )     
    p.add_argument('--save_model_name',                                               type=str,   default='model.pt'                         )                             
    p.add_argument('--save_model_every',                                              type=int,   default=10                                 )                                     
    p.add_argument('--rna_file_name',                                                 type=str,   default='rna.npy'                          )                              
    p.add_argument('--rna_file_suffix',                                               type=str,   default='*FPKM-UQ.txt'                     )                        
    p.add_argument('--embedding_file_suffix_rna',                                     type=str                                               )                        
    p.add_argument('--embedding_file_suffix_image',                                   type=str                                               )                        
    p.add_argument('--embedding_file_suffix_image_rna',                               type=str                                               )                        
    p.add_argument('--rna_file_reduced_suffix',                                       type=str,   default='_reduced'                         )                             
    p.add_argument('--use_unfiltered_data',                                           type=str,   default='True'                             )                                
    p.add_argument('--class_numpy_file_name',                                         type=str,   default='class.npy'                        )                            
    p.add_argument('--wall_time',                                                     type=int,    default=24                                )
    p.add_argument('--seed',                                                          type=int,    default=0                                 )
    p.add_argument('--nn_mode',                                                       type=str,    default='pre_compress'                    )
    p.add_argument('--use_same_seed',                                                 type=str,    default='False'                           )
    p.add_argument('--nn_type_img',                                       nargs="+",  type=str,    default='VGG11'                           )
    p.add_argument('--nn_type_rna',                                       nargs="+",  type=str,    default='DENSE'                           )
    p.add_argument('--hidden_layer_encoder_topology', '--nargs-int-type', nargs='*',  type=int,                                              )                             
    p.add_argument('--encoder_activation',                                nargs="+",  type=str,    default='sigmoid'                         )                              
    p.add_argument('--nn_dense_dropout_1',                                nargs="+",  type=float,  default=0.0                               )                                    
    p.add_argument('--nn_dense_dropout_2',                                nargs="+",  type=float,  default=0.0                               )                                    
    p.add_argument('--dataset',                                                       type=str                                               )
    p.add_argument('--cases',                                                         type=str,    default='ALL_ELIGIBLE_CASES'              )
    p.add_argument('--divide_cases',                                                  type=str,    default='False'                           )
    p.add_argument('--cases_reserved_for_image_rna',                                  type=int                                               )
    p.add_argument('--data_source',                                                   type=str                                               )
    p.add_argument('--global_data',                                                   type=str                                               )
    p.add_argument('--mapping_file_name',                                             type=str,    default='mapping_file'                    )
    p.add_argument('--target_genes_reference_file',                                   type=str                                               )
    p.add_argument('--input_mode',                                                    type=str,    default='NONE'                            )
    p.add_argument('--multimode',                                                     type=str,    default='NONE'                            )
    p.add_argument('--n_samples',                                         nargs="+",  type=int,    default="101"                             )                                    
    p.add_argument('--n_tiles',                                           nargs="+",  type=int,    default="50"                              )       
    p.add_argument('--n_tests',                                                       type=int,    default="16"                              )       
    p.add_argument('--highest_class_number',                              nargs="+",  type=int,    default="989"                             )                                                             
    p.add_argument('--supergrid_size',                                                type=int,    default=1                                 )                                      
    p.add_argument('--patch_points_to_sample',                                        type=int,    default=1000                              )                                   
    p.add_argument('--tile_size',                                         nargs="+",  type=int,    default=128                               )                                    
    p.add_argument('--gene_data_norm',                                    nargs="+",  type=str,    default='NONE'                            )                                 
    p.add_argument('--gene_data_transform',                               nargs="+",  type=str,    default='NONE'                            )
    p.add_argument('--n_genes',                                                       type=int,    default=555                               )                                   
    p.add_argument('--remove_unexpressed_genes',                                      type=str,    default='True'                            )                               
    p.add_argument('--remove_low_expression_genes',                                   type=str,   default='True'                             )                                
    p.add_argument('--low_expression_threshold',                                      type=float, default=0                                  )                                
    p.add_argument('--batch_size',                                        nargs="+",  type=int,   default=64                                 )                                     
    p.add_argument('--learning_rate',                                     nargs="+",  type=float, default=.00082                             )                                 
    p.add_argument('--n_epochs',                                                      type=int,   default=17                                 )
    p.add_argument('--n_iterations',                                                  type=int,   default=251                                )
    p.add_argument('--pct_test',                                          nargs="+",  type=float, default=0.2                                )
    p.add_argument('--final_test_batch_size',                                         type=int,   default=1000                               )                                   
    p.add_argument('--lr',                                                nargs="+",  type=float, default=0.0001                             )
    p.add_argument('--latent_dim',                                                    type=int,   default=7                                  )
    p.add_argument('--l1_coef',                                                       type=float, default=0.1                                )
    p.add_argument('--em_iters',                                                      type=int,   default=1                                  )
    p.add_argument('--clip',                                                          type=float, default=1                                  )
    p.add_argument('--max_consecutive_losses',                                        type=int,   default=7771                               )
    p.add_argument('--optimizer',                                         nargs="+",  type=str,   default='ADAM'                             )
    p.add_argument('--label_swap_perunit',                                            type=float, default=0.0                                )                                    
    p.add_argument('--make_grey_perunit',                                             type=float, default=0.0                                ) 
    p.add_argument('--peer_noise_perunit',                                            type=float, default=0.0                                ) 
    p.add_argument('--regenerate',                                                    type=str,   default='True'                             )
    p.add_argument('--just_profile',                                                  type=str,   default='False'                            )                        
    p.add_argument('--just_test',                                                     type=str,   default='False'                            )                        
    p.add_argument('--rand_tiles',                                                    type=str,   default='True'                             )                         
    p.add_argument('--points_to_sample',                                              type=int,   default=100                                )                            
    p.add_argument('--min_uniques',                                                   type=int,   default=0                                  )                              
    p.add_argument('--min_tile_sd',                                                   type=float, default=3                                  )                              
    p.add_argument('--greyness',                                                      type=int,   default=0                                  )                              
    p.add_argument('--stain_norm',                                        nargs="+",  type=str,   default='NONE'                             )                         
    p.add_argument('--stain_norm_target',                                             type=str,   default='NONE'                             )                         
    p.add_argument('--cancer_type',                                                   type=str,   default='NONE'                             )                 
    p.add_argument('--cancer_type_long',                                              type=str,   default='NONE'                             )                 
    p.add_argument('--class_names',                                       nargs="*",  type=str,   default='NONE'                             )                 
    p.add_argument('--long_class_names',                                  nargs="+",  type=str,   default='NONE'                             ) 
    p.add_argument('--class_colours',                                     nargs="*"                                                          )                 
    p.add_argument('--colour_map',                                                    type=str,   default='tab10'                            )    
    p.add_argument('--target_tile_coords',                                nargs=2,    type=int,   default=[2000,2000]                        )                 
    p.add_argument('--zoom_out_prob',                                     nargs="*",  type=float,                                            )                 
    p.add_argument('--zoom_out_mags',                                     nargs="*",  type=int,                                              )                 

    p.add_argument('--a_d_use_cupy',                                                  type=str,   default='True'                             )                    
    p.add_argument('--cov_threshold',                                                 type=float, default=8.0                                )                    
    p.add_argument('--cov_uq_threshold',                                              type=float, default=0.0                                )                    
    p.add_argument('--cutoff_percentile',                                             type=float, default=0.05                               )                    
 
    p.add_argument('--figure_width',                                                  type=float, default=16                                 )                                  
    p.add_argument('--figure_height',                                                 type=float, default=16                                 )
    p.add_argument('--annotated_tiles',                                               type=str,   default='True'                             )
    p.add_argument('--scattergram',                                                   type=str,   default='True'                             )
    p.add_argument('--box_plot',                                                      type=str,   default='True'                             )
    p.add_argument('--minimum_job_size',                                              type=float, default=5                                  )
    p.add_argument('--probs_matrix',                                                  type=str,   default='True'                             )
    p.add_argument('--probs_matrix_interpolation',                                    type=str,   default='none'                             )
    p.add_argument('--show_patch_images',                                             type=str,   default='True'                             )    
    p.add_argument('--show_rows',                                                     type=int,   default=500                                )                            
    p.add_argument('--show_cols',                                                     type=int,   default=100                                ) 
    p.add_argument('--bar_chart_x_labels',                                            type=str,   default='rna_case_id'                      )
    p.add_argument('--bar_chart_show_all',                                            type=str,   default='True'                             )
    p.add_argument('--bar_chart_sort_hi_lo',                                          type=str,   default='True'                             )
    p.add_argument('-ddp', '--ddp',                                                   type=str,   default='False'                            )  # only supported for 'NN_MODE=pre_compress' ATM (the auto-encoder front-end)
    p.add_argument('-n', '--nodes',                                                   type=int,   default=1,  metavar='N'                    )  # only supported for 'NN_MODE=pre_compress' ATM (the auto-encoder front-end)
    p.add_argument('-g', '--gpus',                                                    type=int,   default=1,  help='number of gpus per node' )  # only supported for 'NN_MODE=pre_compress' ATM (the auto-encoder front-end)
    p.add_argument('-nr', '--nr',                                                     type=int,   default=0,  help='ranking within node'     )  # only supported for 'NN_MODE=pre_compress' ATM (the auto-encoder front-end)
    
    p.add_argument('--hidden_layer_neurons',                              nargs="+",  type=int,    default=2000                              )     
    p.add_argument('--gene_embed_dim',                                    nargs="+",  type=int,    default=1000                              )    
    
    p.add_argument('--use_autoencoder_output',                                        type=str,   default='False'                            ) # if "True", use file containing auto-encoder output (which must exist, in log_dir) as input rather than the usual input (e.g. rna-seq values)
    p.add_argument('--ae_add_noise',                                                  type=str,   default='False'                            )
    p.add_argument('--clustering',                                                    type=str,   default='NONE'                             )
    p.add_argument('--n_clusters',                                                    type=int,                                              )
    p.add_argument('--metric',                                                        type=str,   default="manhattan"                        )        
    p.add_argument('--epsilon',                                                       type=float, default="0.5"                              )        
    p.add_argument('--perplexity',                                        nargs="+",  type=float, default="30"                               )        
    p.add_argument('--momentum',                                                      type=float, default=0.8                                )        
    p.add_argument('--min_cluster_size',                                              type=int,   default=3                                  )        

    args, _ = p.parse_known_args()

    is_local = args.log_dir == 'experiments/example'

    args.n_workers  = 0 if is_local else 12
        
    main(args)














  
