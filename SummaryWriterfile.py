import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
data = []

logdirs = [r'E:\SSL4MIS-master\model\ACDC\Cross_consistency_training_3_labeled\log',
           r'E:\SSL4MIS-master\model\ACDC\Cross_pseudo_supervision_3_labeled\unet\log',
           r'E:\SSL4MIS-master\model\ACDC\Cross_teaching_between_cnn_transformer_3_labeled\unet\log1',
           r'E:\SSL4MIS-master\model\ACDC\Deep_Co_Training_3_labeled\unet\log',
           r'E:\SSL4MIS-master\model\ACDC\Entropy_Minimization_3_labeled\unet\log',
           r'E:\SSL4MIS-master\model\ACDC\Interpolation_Consistency_Training_3_labeled\unet\log',
           r'E:\SSL4MIS-master\model\ACDC\Mean_Teacher_3_labeled\unet\log',
           r'E:\SSL4MIS-master\model\ACDC\Uncertainty_Aware_Mean_Teacher_3_labeled\unet\log',
           r'E:\SSL4MIS-master\model\ACDC\Uncertainty_Rectified_Pyramid_Consistency_3_labeled\log',
           r'E:\SSL4MIS-master\model\ACDC\MyHiFormer_and_Match_cross_pseudo_supervision_3\myHiformer\log1']
keywords = ['val_mean_dice',
           'model1_val_mean_dice',
           'model1_val_mean_dice',
           'val_mean_dice',
           'val_mean_dice',
           'val_mean_dice',
           'val_mean_dice',
           'val_mean_dice',
           'val_mean_dice',
           'model2_val_mean_dice']

for file_path,keyword in zip(logdirs,keywords):
    event_acc = event_accumulator.EventAccumulator(file_path)
    event_acc.Reload()  # 加载数据
    scalar_events = event_acc.Scalars('info/'+keyword)  # scalar_name是你要提取的标量数据的名称
    steps = [event.step for event in scalar_events]
    values = [event.value for event in scalar_events]
    data.append((steps, values))
plt.figure(figsize=(10, 6))
for steps, values in data:
    plt.plot(steps, values)
plt.xlabel("Steps")
plt.ylabel("val_mean_dice")
plt.title("Mean dice of 5% ACDC labeled data training")
plt.legend(['CCT', 'CPS','CTCT', 'DCT','EM','ICT','MT','UA-MT','URPC','Ours'])  # 根据需要修改图例标签
plt.grid(True)
plt.show()
plt.savefig('../features/All_Dice.png')

# labels = ['CCT', 'CPS','CTCT', 'DCT','EM','ICT','MT','Ours','UA-MT','URPC']
# plot_line_chart(steps_list, values_list, labels)

