#!/bin/bash
URL_BASE="https://lmb.informatik.uni-freiburg.de/resources/binaries/net_models/SceneFlow"

download () {
	net=$1
	evo=$2
	state=$3
	subpath="$net/training/$evo/checkpoints"
	wget  "$URL_BASE/$subpath/snapshot-$state.data-00000-of-00001" -P $subpath
	wget  "$URL_BASE/$subpath/snapshot-$state.index" -P $subpath
	wget  "$URL_BASE/$subpath/snapshot-$state.meta" -P $subpath
}

download occ-fill 00__flyingThings3D.train__S_custom 220000
download occ-fill-kitti 00__kitti_sceneflow_ft__S_custom 200000
