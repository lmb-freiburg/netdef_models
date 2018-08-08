# net_models
Repository for different network models related to flow/disparity

# Setup
* Install [tensorflow]()
* Compile [lmbspecialops]()
* Add [netdef_slim]() to your python path

#Running networks

* Clone this repository.
* Change your directory to the network directory (Eg: FlowNet3).
* Run download_snapshots.sh. This takes a while to download all snapshots.
* Now you should be ready to run the networks. Change your directory to a network type (Eg: css).
  Use the following command to test the network on an image pair:
  `python3 controller.py image0_path image1_path out_dir`
