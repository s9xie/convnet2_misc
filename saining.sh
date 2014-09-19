#deep cotrain 1024
#python convnet.py --data-path /workplace/sxie/car_view_mixed_batches --train-range 0-70 --test-range 1000-1002 --save-path /workplace/sxie/convnet2_models --epochs 90 --layer-def layers/sxie/layers-car-view-deep-1gpu.cfg --layer-params layers/sxie/layer-params-car-view-deep-1gpu.cfg --data-provider taskscotr --inner-size 224 --gpu 1 --mini 128 --test-freq 201 --color-noise 0.1 

#deep cotrain no fc
python convnet.py --data-path /workplace/sxie/car_view_mixed_batches --train-range 0-70 --test-range 1000-1002 --save-path /workplace/sxie/convnet2_models --epochs 120 --layer-def layers/sxie/layers-car-view-deep-original.cfg --layer-params layers/sxie/layer-params-car-view-deep-original.cfg --data-provider taskscotr --inner-size 224 --gpu 1 --mini 128 --test-freq 20 --color-noise 0.1 

##deep cotrain 4096
#python convnet.py --data-path /workplace/sxie/car_view_mixed_batches --train-range 0-70 --test-range 1000-1002 --save-path /workplace/sxie/convnet2_models --epochs 90 --layer-def layers/sxie/layers-car-view-deep-bigger-1gpu.cfg --layer-params layers/sxie/layer-params-car-view-deep-bigger-1gpu.cfg --data-provider taskscotr --inner-size 224 --gpu 1 --mini 128 --test-freq 201 --color-noise 0.1 

#naive cotrain
#python convnet.py --data-path /workplace/sxie/car_view_mixed_batches --train-range 0-70 --test-range 1000-1002 --save-path /workplace/sxie/convnet2_models --epochs 90 --layer-def layers/sxie/layers-car-view-1gpu.cfg --layer-params layers/sxie/layer-params-car-view-1gpu.cfg --data-provider taskscotr --inner-size 224 --gpu 1 --mini 128 --test-freq 50 --color-noise 0.1 

#two gpu cotrain
#python convnet.py --data-path /workplace/sxie/car_view_mixed_batches --train-range 0-70 --test-range 1000-1002 --save-path /workplace/sxie/convnet2_models --epochs 90 --layer-def layers/sxie/layers-car-view-2gpu-data.cfg --layer-params layers/sxie/layer-params-car-view-2gpu-data.cfg --data-provider taskscotr --inner-size 224 --gpu 0,1 --mini 256 --test-freq 50 --color-noise 0.1 

#naive cotrain - orginal parameters
#python convnet.py --data-path /workplace/sxie/car_view_mixed_batches --train-range 0-70 --test-range 1000-1002 --save-path /workplace/sxie/convnet2_models --epochs 120 --layer-def layers/sxie/layers-car-view-original.cfg --layer-params layers/sxie/layer-params-car-view-original.cfg --data-provider taskscotr --inner-size 224 --gpu 0 --mini 128 --test-freq 100 --color-noise 0.1 
