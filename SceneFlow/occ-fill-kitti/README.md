## Network for filling disparity values in occluded regions (finetuned on KITTI)

Please download snapshots for FlowNet-CSS-kitti and DispNet-CSS-kitti first.
Example command to generate interpolated output:

```
python3 controller.py eval --imgs_t0 test_images/t0_imgL.png test_images/t0_imgR.png \
                           --imgs_t1 test_images/t1_imgL.png test_images/t1_imgR.png \
                           --dn_path ../../DispNet3/CSS-ft-kitti/ \
                           --fn_path ../../FlowNet3/CSS-ft-kitti/
```

