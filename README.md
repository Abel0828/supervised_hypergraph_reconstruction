# Supervised Hyeprgraph Reconstruction (Data and Code)
Anonymous Authors

## Data Sources
* Enron: https://www.cs.cornell.edu/~arb/data/email-Enron/
* DBLP: https://www.cs.cornell.edu/~arb/data/coauth-DBLP/ (we use 2016 as training, 2015 as query)
* P.School: https://www.cs.cornell.edu/~arb/data/contact-primary-school/
* H.School: https://www.cs.cornell.edu/~arb/data/contact-high-school/
* Foursquare: https://networks.skewed.de/net/foursquare (the `NYC_restaurant_checkin` collection)
* Hosts-Virus: https://zenodo.org/record/807517#.YgSOoerMJdg (the `data/associations.csv` file)
* Directors: https://networks.skewed.de/net/board_directors (the `net2m_2002-05-01` collection)
* Crimes: https://networks.skewed.de/net/crime 

We have processed all data from the sources and put them in `data/`. Notice that the P.School dataset is put in (and renamed) `school`, the H.School dataset is put in (and renamed) `school2`.

## Environment
* `python >= 3.5`, [Anaconda3](https://www.anaconda.com/)
* `numpy>=1.20`
* `sklearn>=0.24`
* `networkx>=2.5.1`
* `graph-tool>=2.44` (for evaluating the baseline Bayesian-MDL)
* `cdlib>=0.2.5`  (for evaluating the baselines Demon, CFinder)
* `tqdm`
* Build:
`g++ -std=c++11 -pthread cmotif.cpp -o cmotif`

## Running the Code
`python main.py --dataset <dataset> --beta <beta> --features <features>`

* `<features>` can be either `count` or `motif`.
* As mentioned in the paper, `beta` depend on datasets. Combinations:
    * `--dataset dblp --beta 1000000`
    * `--dataset enron --beta 1000`
    *  `--dataset school --beta 350000`
    * `--dataset school2 --beta 60000`
    *  `--dataset foursquare --beta 20000`
    * `--dataset hosts --beta 6000`
    *  `--dataset directors --beta 800`
    * `--dataset crime --beta 1000`


