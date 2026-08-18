import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
os.environ["CUDA_VISIBLE_DEVICES"]="1"   

warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    #model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-WTConv.yaml') #tpav2
    #model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-C2TPA.yaml') # TPAv2
    #model = YOLO('/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/11/yolo11-CSFCN.yaml')
    #model = YOLO('/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/11/yolo11-CGRFPN.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    #model = YOLO('ultralytics/cfg/models/11/yolo11-ReCalibrationFPN-P345.yaml')
    #model = YOLO('ultralytics/cfg/models/add/yolo11-c3k2-lglb-c2tpa.yaml')
    #model = YOLO('ultralytics/cfg/models/11/yolo11-C2TSSA-DYT-Mona-SEFN.yaml')
    #model = YOLO("ultralytics/cfg/models/add/yolo11-c2tssa-dyt-mona-sefn-lglb.yaml")
    #model = YOLO("ultralytics/cfg/models/11/yolo11-C3k2-LGLB.yaml")#
    #model = YOLO("/root/project/ultralytics-yolo11-main/yolo11-n-chaofen/Flow-chaofen6/weights/best.pt")
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/add/yolo11-CGRFPN-C2TPA.yaml")
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/11/yolo11-C3k2-ConvFormer.yaml")
    #====#model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/add/yolo11-c2tpa-IIFM.yaml")
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/add/yolo11-c2tpa-IIFM.yaml")
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/11/yolo11-ASF.yaml")
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/add/yolo11-dysample-c2tpa.yaml")
    #model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/add/II-yolo.yaml")
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/add/ii-yolo-n.yaml")
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml")
#========================================================================================================
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/add/yolo11-iifm.yaml")
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/add/yolo11-sml.yaml")
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/add/IIFM+SML.yaml")
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/add/c2tpa+SML.yaml")
    model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/add/II-yolo-new.yaml")#BN
#==========================================================================
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/add/II-yolo.yaml")
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/hyper-yolo/hyper-yolo.yaml")
#==============================================================
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/8/v8-n.yaml")
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/add/II-yolo.yaml")
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/11/yolo11-p2.yaml")
#=====================================================
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/add/C2PSA-SE.yaml")#SE
#================================================
    #model = YOLO("/root/project/ultralytics-yolo11-main/ultralytics/cfg/models/add/yolo11-iifm.yaml")
    model.train(
                #data='/root/project/Flow-img-v11/datasets/AFO/data.yaml',
                #data='Flow.yaml',
		#data = "/root/project/ultralytics-yolo11-main/dataset/FCOS/data.yaml",
		#data = "/root/project/ultralytics-yolo11-main/dataset/floating-new/data.yaml",
		#data = "/root/project/ultralytics-yolo11-main/dataset/yolo_dataset_correct/data.yaml",
		#data = "/root/project/ultralytics-yolo11-main/dataset/newnew/data.yaml",
		data = "/root/project/ultralytics-yolo11-main/dataset/data.yaml",
		#data = "/root/project/Flow-img-v11/datasets/floating-waste-I-enhanced/data.yaml",

                cache=False,
                imgsz=640,
                epochs=120,
                batch=8,
                close_mosaic=10, 
                workers=2,
                #device=0, 
		#device = 1,
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, #
                # amp=False, # 
                # fraction=0.2,
		#hsv_h = 0,
		#hsv_s = 0,
		#hsv_v = 0,
                #project='CSFCN',
                #project='yolo11-C3k2-C2TPAv2',
                #project= 'yolo11-C3k2-WTConv-c2tpav2',
                #project = "yolo11-c3k2-lglb-v2",
                #project = 'tpa_lka_fusion',
                #project = "yolo11-ReCalibrationFPN-P345",
                #project = 'yolov11-n-segFlow',
                #project = "yolo11-lglb-c2tpav2",
                #project = 'yolo11-C2TSSA-DYT-Mona-SEFN',
                #project = "yolo11-n-chaofen",
                #project = "yolo11-CGRFPN",
                #project = "yolo11-CGRFPN-C2TPA",
                #project = "IIFM-asf",
                #project = "New_c2tpa-IIFM",
                #project = "yolo11-dysample-c2tpa",
                #project = "iifm_V3-c2tpa-remove-sppf",
                #project = "mamba-yolo",
                #project = "yolo11-n-p2",
                #project = "yolov8n",
                #project = "final-model-s",
		#project = "hyper-yolo",
		#project = "IIFM-C2TPA",
		#project = "C2TPA+SML",
		#project = "II-YOLO-new",
                #project = "IIFM-MASK0.5",
		project = "II-YOLO",
		#project = "yolo11+IIFM",#0605
                #name='Flow-200epochs',
                #name = "yolo11-C2TSSA-DYT-Mona-SEFN"
                #name = 'yolo11-C2TSSA-DYT-Mona-SEFN-LGLB'
                #name = 'WTConv-c2tpav2'
                #name = "highlight",
                #name = "FCOS"
		#name = "fw-master"
		#name = "floating-waste-I"
		#name = "Flow_RI"
		name = "Garbage"
                )
