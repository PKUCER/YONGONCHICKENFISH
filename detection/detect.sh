# python detect.py --save-txt --conf 0.1 --img 608 \
# --output inference/testimgs --source testimgs/ \
# --cfg cfg/satellite-anchor.cfg --name cfg/satellite.names \
# --weights runs/exp39_test-allanno-350resume/weights/best_test-allanno-350resume.pt

python detect.py --save-txt --conf 0.02 --img 608 \
--output inference/2010 \
--source detect_buffer_jpg_2010/ \
--cfg cfg/satellite-anchor-608.cfg --name cfg/satellite.names \
--weights runs/exp64_0829-no-intersect-608-new-new/weights/best_0829-no-intersect-608-new-new.pt

python detect.py --save-txt --conf 0.02 --img 608 \
--output inference/2011 \
--source detect_buffer_jpg_2011/ \
--cfg cfg/satellite-anchor-608.cfg --name cfg/satellite.names \
--weights runs/exp64_0829-no-intersect-608-new-new/weights/best_0829-no-intersect-608-new-new.pt

python detect.py --save-txt --conf 0.02 --img 608 \
--output inference/2012 \
--source detect_buffer_jpg_2012/ \
--cfg cfg/satellite-anchor-608.cfg --name cfg/satellite.names \
--weights runs/exp64_0829-no-intersect-608-new-new/weights/best_0829-no-intersect-608-new-new.pt

python detect.py --save-txt --conf 0.02 --img 608 \
--output inference/2013 \
--source detect_buffer_jpg_2013/ \
--cfg cfg/satellite-anchor-608.cfg --name cfg/satellite.names \
--weights runs/exp64_0829-no-intersect-608-new-new/weights/best_0829-no-intersect-608-new-new.pt

python detect.py --save-txt --conf 0.02 --img 608 \
--output inference/2014 \
--source detect_buffer_jpg_2014/ \
--cfg cfg/satellite-anchor-608.cfg --name cfg/satellite.names \
--weights runs/exp64_0829-no-intersect-608-new-new/weights/best_0829-no-intersect-608-new-new.pt

python detect.py --save-txt --conf 0.02 --img 608 \
--output inference/2015 \
--source detect_buffer_jpg_2015/ \
--cfg cfg/satellite-anchor-608.cfg --name cfg/satellite.names \
--weights runs/exp64_0829-no-intersect-608-new-new/weights/best_0829-no-intersect-608-new-new.pt

python detect.py --save-txt --conf 0.02 --img 608 \
--output inference/2016 \
--source detect_buffer_jpg_2016/ \
--cfg cfg/satellite-anchor-608.cfg --name cfg/satellite.names \
--weights runs/exp64_0829-no-intersect-608-new-new/weights/best_0829-no-intersect-608-new-new.pt

python detect.py --save-txt --conf 0.02 --img 608 \
--output inference/2018 \
--source detect_buffer_jpg_2018/ \
--cfg cfg/satellite-anchor-608.cfg --name cfg/satellite.names \
--weights runs/exp64_0829-no-intersect-608-new-new/weights/best_0829-no-intersect-608-new-new.pt

# python detect_new.py --conf 0.02 --img 608 --output inference/no_water \
# --source imglist/val_nowater.txt --save-img \
# --cfg cfg/satellite-anchor.cfg --name cfg/satellite.names \
# --weights weights/V2_best_608.pt