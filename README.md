# PIA

**PIA - Protein Interaction Analyzer**

Extract protein-ligand interactions and their frequencies to score and predict the activity of a complex.

## Docker Usage

Either manually build the image using the provided Dockerfile:

```bash
docker image build -f Dockerfile -t michabirklbauer/pia:latest .
```

OR pull it from dockerhub via [michabirklbauer/pia:latest](https://hub.docker.com/r/michabirklbauer/pia):

```bash
docker pull michabirklbauer/pia:latest
```

Once the image is built or downloaded you should create a directory that can be mounted to the container to share files e.g. in Windows 10 create a directory on drive `C:` called `docker_share`. You will have to add this directory in Docker *Settings > Resources > File Sharing* and restart Docker. Then the container can be run with:

```bash
docker run -v C:\docker_share:/exchange -p 8888:8888 -it michabirklbauer/pia:latest
```

This will mount the directory `C:/docker_share` to the directory `/exchange` in the container and files can be shared freely by copy-pasting into those directories. Python can be run normally in the container and PIA can be imported without needing to install any requirements. To get started copy `PIAScript.py` and the files you want to process into `C:/docker_share`. Navigate to `/exchange` in the container to run an analysis. For example, to build a scoring model we would do the following (files used are provided in the `example_files` directory):

```bash
# navigate to the exchange directory containing all files from docker_share
cd exchange
# run a scoring workflow
python3 PIAScript.py -m score -f 6hgv.pdb sEH_6hgv_results.sdf
# inspect training summary
cat sEH_6hgv_results*.txt
```

This will create scoring models, evaluation files and a summary of quality metrics in the `/exchange` and `C:/docker_share` directory.

## Contact

- Mail: [micha.birklbauer@gmail.com](mailto:micha.birklbauer@gmail.com)
- Telegram: [https://telegram.me/micha_birklbauer](https://telegram.me/micha_birklbauer)
