Introduction: 
For constrained end devices in Internet of Things (IoT), such as smart meters, data transmission is an energy-consuming operation. To address this problem, we propose an efficient and privacy-preserving aggregation system with the aid of Fog computing architecture, named PPFA, which enables the intermediate Fog nodes to periodically collect data from nearby smart meters and accurately derive aggregate statistics as the fine-grained Fog level aggregation. The Cloud/utility supplier computes overall aggregate statistics by aggregating Fog level aggregation. To minimize the privacy leakage and mitigate the utility loss, we use more efficient and concentrated Gaussian mechanism to distribute noise generation among parties, thus offering provable differential privacy guarantees of the aggregate statistic on both Fog level and Cloud level. In addition, to ensure aggregator obliviousness and system robustness, we put forward a two-layer encryption scheme: the first layer applies OTP to encrypt individual noisy measurement to achieve aggregator obliviousness, while the second layer uses public-key cryptography for authentication purpose. Our scheme is simple, efficient and practical, it requires only one round of data exchange among a smart meter, its connected Fog node and the Cloud if there are no node failures, otherwise, one extra round is needed between a meter, its connected Fog node and the trusted third party.

How to run:
To reproduce the results in "PPFA: Privacy Preserving Fog-enabled Aggregation in Smart Grid", run the below command:
PPFA_gau
data source: http://www.ucd.ie/issda/data/commissionforenergyregulationcer/

Requirements:
Matlab

Remember to cite the following papers if you use any of the code:
@article{lyu2018ppfa,
  title={PPFA: Privacy Preserving Fog-enabled Aggregation in Smart Grid},
  author={Lyu, Lingjuan and Nandakumar, Karthik and Rubinstein, Benjamin and Jin, Jiong and Bedo, Justin and Palaniswami, Marimuthu},
  journal={IEEE Transactions on Industrial Informatics},
  year={2018},
  publisher={IEEE}
}