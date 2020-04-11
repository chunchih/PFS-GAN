# PFS-GAN
* Training
 * Baselines Training
  * BaselineS: python baselineS.py --dataset_dir <folder_path> --gpu 0
  * BaselineT: python baselineT.py --dataset_dir <folder_path> --pretrained_model <generator_path> --gpu 0 
  * CoGAN: python CoGAN.py --dataset_dir_S <source_dataset_path> --dataset_dir_T <target_dataset_path> --gpu 0 
  * UNIT: 
 * PFS-GAN Training
  * Stage1 Training: python stage1.py --dataset_dir <source_dataset_path> --gpu 0 
  * Copy 'gen_#', 'enc_c_#' into root foler.
  * PFS-GAN Training: python PFS-GAN.py --train_dataset <training_target_dataset> --test_dataset <testing_source_dataset> --source_dataset <source_dataset> --model_name <# of model> --recon_ratio <recon_ratio> --gan_ratio <gan_ratio> --relation_ratio <relation_ratio> --gpu 0

