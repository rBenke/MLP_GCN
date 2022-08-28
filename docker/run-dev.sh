sudo docker run --gpus all --rm -it \
	-v /home/rbenke/repo/mlpVSgcn:/project \
	--name mlp_gcn \
	mlp_gcn /bin/bash
