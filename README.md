# cluster_templates

Instructions on using cluster CS4921

You need to be on the NPS VPN and you will use your NPS credentials to ssh into submission node

```ssh nps_username@cs4921-submit.ern.nps.edu```

once on the node you can log in the following manner to the compute node

```srun --pty --mem=32G --gres=gpu:1 --partition=cs4921 bash```

it is important to specify partition!


In order to use conda you need to load language support:

lists all packages command: 

```module avail```

in order to load conda:

```module load lang/miniconda3/4.8.3```    


to make your conda enviroment:

Create a conda environment and install most common packages 
use anaconda if you want minimalist instalation dont specify anaconda

```conda create --name environment-name python=3.X``` 

source activate environment-name

```conda install tensorflow-gpu=2.2.0```

```conda list```        #list of packages used for the environment


to list all of the conda enviroments that you have:

```conda env list```

To export environment file: 

```activate environment-name```

```conda env export > environment-name.yml```

For other person to use the environment:

```conda env create -f environment-name.yml```


Setting up SSH is important in order to bring your code onto the cluster.
follow instructions on giltab page:

https://docs.gitlab.com/ee/ssh/



following ssh command will verify setup by printing out a welcome msg!

```ssh -T git@gitlab.nps.edu```

SLURM job submission:

```sbatch your_SBATCH.sh```

Every .sh file when made needs to get the right level of permissions

    ```chmod +x your_SBATCH.sh```

List of useful commands for SLURM:

