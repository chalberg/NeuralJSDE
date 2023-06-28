# LatentSDE
Implementation of Latent SDE models in PyTorch for performing inference with informatively sampled, irregularly observed, and sparsely recorded Electronic Health Record data.

---
## Installation
Create a working directory and clone this library using the following command:

```shell script
git clone https://github.com/chalberg/LatentSDE.git
```
Ensure the following requirements and packages are installed and up to date:
* Python >= 3.6
* PyTorch >= 1.6.0
* google-cloud-bigquery
* pandas-gbq
* pandas

You can run the following line to install all these requirements using pip:

```shell script
pip install -r LatentSDE/requirements.txt
```
### Accessing the MIMIC-IV dataset with Google Bigquery
1. To gain access to the MIMIC-IV dataset, visit the [MIMIC-IV project page](https://physionet.org/content/mimiciv/2.2/) and fulfill all the requirements listed under "Files."
2. Once you have been granted access to the dataset, follow the [cloud configuration tutorial](https://mimic.mit.edu/docs/gettingstarted/cloud/) to create a Google Cloud project and link it to your physionet account.
3. Identify the Project ID for your Google Cloud project and use it to set the "project" varible when running  scripts. For example:
```shell script
python main.py --project=[Your Project ID] --model=latent_sde ...
```
4. Follow the pop-up prompt and enter the information for the Google Cloud account used for the BigQuery project.

---
## Usage
**Replicate experiments**

To replicate the experiments in the documentation, run the following command:
```shell script
python main.py --project=[your Project ID]
```
**Save and load data**

To save the cohorts used in the experiments to .csv files, specify a path where you want to save the data and run the following command. Set the "cohort" variable to the dataset you wish to download.

```shell script
python mimic_data.py --data_path=[path to save data] --project=[your Project ID] --cohort=[cohort to download]
```
> Note: This will download the cohort data to the specified directory. Since the MIMIC-IV data may only be shared among authorized users, it is the responsibility of the user to ensure the downloaded data is protected in accordance with the [data use agreement](https://physionet.org/content/mimiciv/view-dua/2.2/).

**Example usage**

This [notebook](example.ipynb) covers example usage of how to
* Query MIMIC-IV using Google BigQuery.
* Save and load the results of a query using functionality from this library.
* Specify and train a latent SDE model for time series inference.

---
## Documentation

For full details of the models, loss, and optimization see the documentation provided [here](documentation.pdf). The following gives a summary of the models and loss objective implemented in this library.

**Models**

We implement latent variable models of the type where $Z(t)$ is some low-dimensional continuous latent process which generates the observable process $Y(t)$ via a contiuous normalizing flow:
$$Y(t) = F(Z(t), O(t), t; \theta),$$
where $O(t)$ may be either a standard Brownian Motion process or Orenstein-Uhlenbeck process of the same dimension as $Z(t)$.

We model the latent process with a Stochastic Differential Equation (SDE) of the form:
$$dZ(t) = d\mu(Z(t),t;\phi) dt + d\sigma(Z(t),t;\rho) dB(t),$$
where the drift $\mu$ and diffusion $\sigma$ are given as deep neural networks with parameters $\phi$ and $\rho$, respectively. We also introduce a prior over the latent process, which is induced by another SDE with the same diffusion function.

**Loss and Optimization**

Given a sequence of observations for an individiual in the dataset $y_{t_1}, y_{t_2}, \ldots, y_{t_n}$, we follow the variational inference framework and seek to maximize the following Evidence Lower Bound (ELBO):
$$\log P(y_{t_1}, y_{t_2}, \ldots, y_{t_n}) \geq \mathbb{E}\left[\sum_{i=1}^n \log P(Y(t_i)|Z(t_i)) - \int_{t_1}^{t_n}\frac{1}{2} ||u(Z(t),t)||_2^2\,dt\right].$$
The function $u(Z(t),t)$ can be thought of as a process which computes the instantaneous KL-divergence between the approximate posterior and prior processes, which are each given by SDEs.
Because of the continuous time nature of the models, we compute gradients of loss with respect to model parameters with the Stochastic Adjoint Sensitivity Method, as introduced in [1].

---
## References

\[1\] Xuechen Li, Ting-Kam Leonard Wong, Ricky T. Q. Chen, David Duvenaud. "Scalable Gradients for Stochastic Differential Equations". *International Conference on Artificial Intelligence and Statistics.* 2020. [[arXiv]](https://arxiv.org/pdf/2001.01328.pdf)

