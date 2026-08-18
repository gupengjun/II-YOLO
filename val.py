import warnings

warnings.filterwarnings('ignore')
import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info


def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'


if __name__ == '__main__':
    #model_path = "/root/project/ultralytics-yolo11-main/mamba-yolo/Flow/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/hyper-yolo/floating-waste-I4/weights/best.pt"
    # model_path = '/root/project/ultralytics-yolo11-main/yolo11-C3k2-WTConv/Flow2/weights/best_fp32.pt'
    # model_path = '/root/project/ultralytics-yolo11-main/yolo11-C3k2-C2TPAV2/Flow/weights/best.pt'
    #model_path = "/root/project/ultralytics-yolo11-main/yolo11-n/Flow3/weights/best.pt"
    # model_path = '/root/project/ultralytics-yolo11-main/yolo11-C3k2-C2TPAv2/Flow/weights/best.pt'
    # model_path = '/root/project/ultralytics-yolo11-main/yolo11-C3k2-WTConv/Flow4/weights/best.pt'
    # model_path = "/root/project/ultralytics-yolo11-main/yolo11-ReCalibrationFPN-P345/Flow/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/yolo11-lglb-c2tpav2/Flow/weights/best.pt"
    # model_path = '/root/project/ultralytics-yolo11-main/yolo11-C2TSSA-DYT-Mona-SEFN.yaml/yolo11-C2TSSA-DYT-Mona-SEFN.yaml/weights/best.pt'
    # model_path = '/root/project/ultralytics-yolo11-main/yolo11-C2TSSA-DYT-Mona-SEFN/yolo11-C2TSSA-DYT-Mona-SEFN-LGLB/weights/best.pt'
    # model_path = '/root/project/ultralytics-yolo11-main/yolo11-C3k2-WTConv-c2tpav2/WTConv-c2tpav22/weights/best.pt'
    # model_path = "/root/project/ultralytics-yolo11-main/yolo11-c3k2-lglb-v2/Flow3/weights/best.pt"
    # model_path = '/root/project/ultralytics-yolo11-main/tpa_lka_fusion/Flow/weights/best.pt'
    # model_path = "/root/project/ultralytics-yolo11-main/yolo11-CGRFPN/Flow2/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/yolo11-n-chaofen/Flow-chaofen6/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/yolo11-CGRFPN-C2TPA/Flow2/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/yolo11-n/Flow2/weights/best.pt"#chaofenbianlv
    # model_path = '/root/project/ultralytics-yolo11-main/yolo11-n/Flow3/weights/best.pt'
    # model_path = "/root/project/Flow-img-v11/Flow-img-yolo11-c2tpa/yolo11-c2tpa2/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/yolo11-C3k2-LGLB-C2TPA/Flow3/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/yolo11-CGRFPN-C2TPA/Flow2/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/tpa_lka_fusion/Flow/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/yolo11-dysample/Flow/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/IIFM-asf/Flow/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/iifm-remove-sppf/Flow2/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/c2tpa-IIFM-ASF/Flow10/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/yolo11-IIFM/Flow/weights/best.pt"
    # ================v8===================================================================
    #model_path = "/root/project/ultralytics-yolo11-main/yolov8n/Flow/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/yolo11-n/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/yolo11-s/Flow/weights/best.pt"
    # ==============================================================================
    # model_path = '/root/project/ultralytics-yolo11-main/final-model/Flow2/weights/best.pt'
    # model_path = "/root/project/ultralytics-yolo11-main/final-model/Flow3/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/yolo11-s/Flow/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/yolov8n/highlight/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/yolo11-n/highlight/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/final-model-n/highlight/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/mamba-yolo/highlight/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/final-model-s/highlight/weights/best.pt"
    # =================================================================
    #model_path = "/root/project/ultralytics-yolo11-main/hyper-yolo/floating-waste-I5/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/YOLO11-IIFM/Flow/weights/best.pt"
    #model_path ="/root/project/ultralytics-yolo11-main/YOLO11-SML/Flow/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/IIFM-SML/Flow2/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/final-model-s/Flow/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/C2TPA+SML/Flow/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/C2TPA+SML/Flow4/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/IIFM-C2TPA/floating-waste-I17/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/yolo11-s/Flow/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/II-YOLO-BN/Flow2/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/II-YOLO/LR_DETR9/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/II-YOLO-new/floating-waste-I2/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/SE/Flow/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/ECA/Flow/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/CoordAttention/Flow/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/EMA/Flow/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/rank16/Flow2/weights/best.pt"
#=======================================================================================
    #model_path = "/root/project/ultralytics-yolo11-main/final-model-s/floating-waste-I2/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/mamba-yolo/floating-waste-I/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/final-model-s/floating-waste-I8/weights/best.pt"
    # ===================================================================================
    # model_path = "/root/project/ultralytics-yolo11-main/iifm-c2tpa-remove-sppf/Floatingwaste2/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/iifm-c2tpa-remove-sppf/Flow4/weights/best.pt"
    # model_path = "/root/project/ultralytics-yolo11-main/yolo11-n/highlight/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/hyper-yolo/floating-waste-I/weights/best.pt"
#==============================================================
    #model_path = "/root/project/ultralytics-yolo11-main/II-YOLO/AFO6/weights/best.pt"
    #model_path = "/root/project/ultralytics-yolo11-main/yolo11+IIFM/Flow8/weights/best.pt"
    model_path = "/root/project/ultralytics-yolo11-main/II-YOLO/Garbage/weights/best.pt"
    model = YOLO(model_path) 
    result = model.val(#data='Flow.yaml',
                   data='/root/project/ultralytics-yolo11-main/dataset/data.yaml',
                   # data='highlight.yaml',
                   #data = "/root/project/Flow-img-v11/datasets/LRDETR/data.yaml",
		   #data = "/root/project/Flow-img-v11/datasets/floating-waste-I-enhanced/data.yaml",
                   split='test',  
                   imgsz=640,
                   batch=4,
                   # iou=0.7,
                   rect=False,
                   device='0',
                   save_json=True, # if you need to cal coco metrice
                   project='runs/test',
                   name='II-YOLOM/Garbage',
                   # name = "yolo11-c2tpa"
                   # name = 'yolov11-c2tpav2-wtconv',
                   #name = 'yolo11+IIFM/Flow2'
                   # name = "yolo11-ReCalibrationFPN-P345"
                   # name = "yolo11-lglb-c2tpa"
                   # name = 'yolo11-lglb-c2tpav2-200epochs'
                   # name = 'yolo11-C2TSSA-DYT-Mona-SEFN'
                   # name = 'yolo11-C2TSSA-DYT-Mona-SEFN-lglb'
                   # name = "WTConv-c2tpav22"
                   # name = 'lglb-v2'
                   # name = "tpa_lka_fusion"
                   #name="YOLO11-IIFM-floatingwaste"
                   #name = "IIFM+SML"
                   # name  = 'yolov11n-highlight'
                   # name = "yolo11-CGRFPN-C2TPA"
                   # name = "c2tpa-TLF/Flow3/"
                   # name = "yolo11-dysample-c2tpa/Flow"
                   # name = "c2tpa-IIFM-ASF"
                   # name = "new_IIFm_c2tpa_remove-sppf"
                   # name = "iifmV3-c2tpa-remove-sppf"
                   #name ="final-s"
                   #name = "C2tpa+SML"
                   #name = "II-YOLO-new"
                   # name = "mamba-yolo"
		   #name = "SE"
		   #name = "ECA"
		   #name = "CoordAttention"
		   #name = "Mask1"
		   #name = "rank8"
		   #name="II-yolo-AFO"
                   )

if model.task == 'detect':  
    length = result.box.p.size
    model_names = list(result.names.values())
    preprocess_time_per_image = result.speed['preprocess']
    inference_time_per_image = result.speed['inference']
    postprocess_time_per_image = result.speed['postprocess']
    all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image

    n_l, n_p, n_g, flops = model_info(model.model)



    model_info_table = PrettyTable()
    model_info_table.title = "Model Info"
    model_info_table.field_names = ["GFLOPs", "Parameters", "前处理时间/一张图", "推理时间/一张图", "后处理时间/一张图", "FPS(前处理+模型推理+后处理)",
                                    "FPS(推理)", "Model File Size"]
    model_info_table.add_row([f'{flops:.1f}', f'{n_p:,}',
                              f'{preprocess_time_per_image / 1000:.6f}s', f'{inference_time_per_image / 1000:.6f}s',
                              f'{postprocess_time_per_image / 1000:.6f}s', f'{1000 / all_time_per_image:.2f}',
                              f'{1000 / inference_time_per_image:.2f}', f'{get_weight_size(model_path)}MB'])
    print(model_info_table)

    model_metrice_table = PrettyTable()
    model_metrice_table.title = "Model Metrice"
    model_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
    for idx in range(length):
        model_metrice_table.add_row([
            model_names[idx],
            f"{result.box.p[idx]:.4f}",
            f"{result.box.r[idx]:.4f}",
            f"{result.box.f1[idx]:.4f}",
            f"{result.box.ap50[idx]:.4f}",
            f"{result.box.all_ap[idx, 5]:.4f}",  # 50 55 60 65 70 75 80 85 90 95
            f"{result.box.ap[idx]:.4f}"
        ])
    model_metrice_table.add_row([
        "all(平均数据)",
        f"{result.results_dict['metrics/precision(B)']:.4f}",
        f"{result.results_dict['metrics/recall(B)']:.4f}",
        f"{np.mean(result.box.f1[:length]):.4f}",
        f"{result.results_dict['metrics/mAP50(B)']:.4f}",
        f"{np.mean(result.box.all_ap[:length, 5]):.4f}",  # 50 55 60 65 70 75 80 85 90 95
        f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
    ])
    print(model_metrice_table)

    with open(result.save_dir / 'paper_data.txt', 'w+', errors="ignore", encoding="utf-8") as f:
        f.write(str(model_info_table))
        f.write('\n')
        f.write(str(model_metrice_table))

    print('-' * 20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-' * 20)
    print('-' * 20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-' * 20)
    print('-' * 20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-' * 20)
    print('-' * 20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-' * 20)
    print('-' * 20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-' * 20)
