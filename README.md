# SML:Skeleton-Based Multi-Feature Learning Method for Sign Language Recognition (SL-GCN)
## Data preparation
1. Download [AUTSL](http://chalearnlap.cvc.uab.es/dataset/40/description/) and [WLASL2000](https://dxli94.github.io/WLASL/) dataset following their instructions.

2. Extract whole-body keypoints data following the instruction in ../data_process/wholepose

3. Run the following code to prepare the data for GCN.
        cd data_gen/
        python autsl_gendata.py
        python gen_bone_data.py
        python gen_motion.py

## Usage
### Train:
```
python main.py --config config/train_joint.yaml

python main.py --config config/train_bone.yaml

python main.py --config config/train_joint_motion.yaml

python main.py --config config/train_bone_motion.yaml

python main.py --config config/train_joint_joint_motion.yaml

python main.py --config config/train_bone_bone_motion.yaml

When loading the dataset, please refer to the feeder_args used in the config file 
and make changes in the corresponding location of the main.py file and adjust the inputs to the model. 
Different networks can be used for training.

If you are not using the MFA module please comment out this line of code in the network's forward:
"self.mm_fusion = MFA(in_channels * 2, 16)"
```

```

### Multi-stream ensemble:
1. Copy the results .pkl files from all streams (joint, bone, joint motion, bone motion joint-joint motion and bone-bone motion) to ensemble/ and renamed them correctly.
2. Follow the instruction in ensemble/ to obtained the results of multi-stream ensemble.
