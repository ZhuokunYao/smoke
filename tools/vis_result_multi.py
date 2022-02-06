import os
import json
import matplotlib.pyplot as plt

OUTPUT_DIR=[
"./checkpoint/waymo_resnet18_768x512_origin",
"./checkpoint/waymo_resnet18_768x512_conner",
"./checkpoint/waymo_resnet18_768x512_bev_iou",
]

#color=['lightgreen', 'red', "blue", "cyan", "darkgreen", "yellow"]
color=['blue', 'red', "darkgreen"]

linestyle=['dashed', ':', '-.', "--", "-", '-']
#marker=['o', 's', '*', "x", "v", '^']
marker=['o', 'o', 'o']
legend=('origin_smoke',
        'v1',
        'v2',)


val_mAP = []
val_step = []
for idx, out_dir in enumerate(OUTPUT_DIR):
    with open(os.path.join(out_dir, "val_mAP.json")) as f:
        val_result = json.load(f)
        if isinstance(val_result,list):
            val_mAP.append(val_result)
            val_step.append([(i+1) * 2000 for i in range(len(val_result))])
        else:
            #val_mAP.append(val_result["val_mAP"])
            val_mAP.append(val_result["range[0,30]"][0])
            val_step.append(val_result["val_step"])

### show
# step2：手动创建一个figure对象，相当于一个空白的画布
figure = plt.figure()

# step3：在画布上添加一个坐标系，标定绘图位置
axes1 = figure.add_subplot(1, 1, 1)
axes1.set_xlabel("step", fontsize=16)#添加x轴坐标标签，后面看来没必要会删除它，这里只是为了演示一下。
axes1.set_ylabel("car's AP@0.5",fontsize=16)#添加y轴标签，设置字体大小为16，这里也可以设字体样式与颜色

# step4：画图,设置线条颜色、线型、点标记符
for idx in range(len(OUTPUT_DIR)):
    axes1.plot(val_step[idx], val_mAP[idx], color=color[idx], linestyle=linestyle[idx], marker=marker[idx])

# step5：展示
plt.legend(legend, loc='upper left')
plt.show()
plt.savefig("tools/vis.png")