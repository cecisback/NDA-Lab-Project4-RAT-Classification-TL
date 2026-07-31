# Project 4: What Network Am I Using? RAT Classification with Transfer Learning
This project was realized for the course of Network Measurements and Data Analysis at Polimi in 2026.

Even if it was thought to be a project group and we managed together every aspect of the project development, I decided to keep and publish in this repository only files and folders that were strictly written by me.

I decided to unify the core functionalities implemented in the Python files located in the src folder in these notebooks, since I think that this is the most efficient way to evaluate the outcome of each step of the classification pipeline. 
The purpose of the project is performing *RAT classification* to predict the network to which the mobile phone is attached even in case of obfuscation. In the project specifications, it is said that the network indicator isn't visible due to dead pixel cluster.

Traffic classification, in our context, is applied to a specific scenario.
The first step that should be performed is *traffic capture* to retrieve measurement data. 
The most known way, easily accessible to all computer devices, to intercept packets across the network relies on *passive observation*.
*Active measurement techniques*, instead, are useful for retrieving insights about the physical and logical path followed by a given packet to reach its destination and coming back to its sender.
In the most complex scenarios and with a strong knowledge of protocols that enable communication over Wireless networks, *Wifi sniffing* is greedily ammissible too. 

## Dataset
Our project relies on a *dataset* extracted from a research paper focused on evaluating *QoS metrics* across *multiple RATs in four different countries*.  

By comparing the given dataset with the ones seen during the evaluation labs, where measurement data were captured with the methods mentioned above, I noticed that the measurement parameters taken into account should reflect:
- the purpose of the traffic capture;
- the network in which data are forwarded;
- the way in which the intended device handles communications over the network;

Due to these criteria, RSS and SSID, which are highly informative in a wireless network, become meaningless in a wired one, since this latter doesn't rely on access points for packets forwarding and interconnection of devices across the network, but rather on routing protocols (such as OSPF and BGP) and on parameters, like TTL and hop count, for traffic analysis purposes.

## Dataset setup
- Download the dataset at: [10.5281/zenodo.15420422;](https://zenodo.org/records/15420422);
- Rename the zip folder as 'dataset.zip';
- Locate the renamed folder into a newly defined path within the cloned repository: data/raw;

## Project Structure

```
.
| - .gitignore
| - README.md
| - requirements.txt
| - data/
	| - analysis/
		| - nan_analysis.txt
	| - outcome_preprocess/
		| - encoded.csv
		| - feature_dataset.csv
	| - raw/                                           (git-ignored — folder dataset.zip with .csv file)
		| - dataset.zip 
		| - *.csv
		| - README.md
	| - node_info.csv
| - notebooks/                                            Jupyter notebooks for exploration & visualisation
	| - .gitkeep
	| - RandomForest_baseline.ipynb
	| - NN_RAT_classification.ipynb
	| - Outcome_visualization.ipynb
| - results/
	| - exported/
		| - aggregated_features_NN.csv
		| - aggregated_features_RF.csv
		| - MLPClassifier.onnx
		| - MLPClassifier.pkl
		| - MLPClassifier.png
		| - outcome_windowed_features_NN.json
		| - outcome_windowed_features_RF.json
	| - figures/          
		| - .gitkeep                                           Plots and graphics
		| - classification_NN/
			| - classification_NN_conf_matrix_{z_value}_normalized.png
			| - classification_NN_conf_matrix_{z_value}.png
		| - classification_RF
			| - classification_NN_conf_matrix_{z_value}_normalized.png
			| - classification_NN_conf_matrix_{z_value}.png
		| - metric_eval
			| - {metric}_vs_z.png
	| - metrics/                                             Evaluation metrics (CSV / JSON)
		| - .gitkeep
		| -	performance_eval_results.json
	| - src/                                        Python source modules (preprocessing, features, models, utils)
		| - __init__.py
		| - utils.py 
		| - utils_eval.py
		| - utils_train_models.py
		| - config.py
		| - preprocessing.py
		| - window_features.py
		| - config.py
```