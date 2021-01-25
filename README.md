# DialogueNLP
This is a PyTorch implementation of the Committed paper for ACL 2021.

### 1 Preparation for Dataset
In order to facilitate project management, the data sets used in this project are linked as [Download]( https://pan.baidu.com/s/1Qt-NSi9akM00s3C3fIVH_g) with password: ev80. 

After the dataset is downloaded, it needs to be decompressed into this project. The extracted file is named dataset, which contains two Chinese datasets mentioned in this paper: Double and Weibo. The link of English datasets is as follows: [Cornell Movie-Dialogu Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

### 2 Models
The proposed model is consist of a Retrieval Model and a Generation model. In order to align utterance and retrieval pairs at word-vector level, a novel Selective-Attention Guided Alignment module is proposed.

### 3 Training
This project is supported by Pytorch and other standard libraries. If you want to train the whole model, *train.ipynb* is a jupyter file, which can directly run and view the intermediate results in the training process.


### Performance
We evaluted the proposed model on BLEU, Rouge, Relevance and Diversity, which the former metrics are word-overlap metrics and the latter two are embedding based metrics. The result are listed as follow:  

<table>
<thead>
  <tr>
    <th rowspan="9">Cornell</th>
    <th colspan="2">Types</th>
    <th rowspan="2">Models</th>
    <th rowspan="2">BLEU</th>
    <th colspan="3">Rouge</th>
    <th colspan="3">Relevance</th>
    <th colspan="2">Diversity</th>
  </tr>
  <tr>
    <td>Rtrv</td>
    <td>Gene</td>
    <td>R-1</td>
    <td>R-2</td>
    <td>R-L</td>
    <td>Average</td>
    <td>Extrema</td>
    <td>Greedy</td>
    <td>Dist-1</td>
    <td>Dist-2</td>
  </tr>
  <tr>
    <td></td>
    <td>√ </td>
    <td>S2S+Attn</td>
    <td>38.79</td>
    <td>37.82</td>
    <td>17.87</td>
    <td>33.73</td>
    <td>0.361</td>
    <td>0.201</td>
    <td>0.346</td>
    <td>0.049</td>
    <td>0.088</td>
  </tr>
  <tr>
    <td></td>
    <td>√ </td>
    <td>CVAE</td>
    <td>45.23</td>
    <td>41.89</td>
    <td>20.86</td>
    <td>39.49</td>
    <td>0.381</td>
    <td>0.256</td>
    <td>0.374</td>
    <td>0.076</td>
    <td>0.145</td>
  </tr>
  <tr>
    <td></td>
    <td>√ </td>
    <td>UniLM</td>
    <td>50.65</td>
    <td>44.24</td>
    <td>23.07</td>
    <td>40.27</td>
    <td>0.401</td>
    <td>0.294</td>
    <td>0.387</td>
    <td>0.121</td>
    <td>0.189</td>
  </tr>
  <tr>
    <td>√ </td>
    <td></td>
    <td>Retrieval</td>
    <td>36.91</td>
    <td>30.81</td>
    <td>13.87</td>
    <td>27.33</td>
    <td>0.296</td>
    <td>0.152</td>
    <td>0.311</td>
    <td>0.103</td>
    <td>0.249</td>
  </tr>
  <tr>
    <td>√ </td>
    <td></td>
    <td>Retrieval+Rerank</td>
    <td>38.67</td>
    <td>34.57</td>
    <td>18.23</td>
    <td>32.44</td>
    <td>0.338</td>
    <td>0.193</td>
    <td>0.321</td>
    <td>0.129</td>
    <td>0.212</td>
  </tr>
  <tr>
    <td>√ </td>
    <td>√ </td>
    <td>Eidt</td>
    <td>49.11</td>
    <td>45.81</td>
    <td>21.99</td>
    <td>43.01</td>
    <td>0.393</td>
    <td>0.307</td>
    <td>0.391</td>
    <td>0.112</td>
    <td>0.207</td>
  </tr>
  <tr>
    <td>√ </td>
    <td>√ </td>
    <td>Ours</td>
    <td>52.67</td>
    <td>48.72</td>
    <td>24.45</td>
    <td>43.28</td>
    <td>0.417</td>
    <td>0.331</td>
    <td>0.407</td>
    <td>0.121</td>
    <td>0.231</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="9">Douban</td>
    <td colspan="2">Types</td>
    <td rowspan="2">Models</td>
    <td rowspan="2">BLEU</td>
    <td colspan="3">Rouge</td>
    <td colspan="3">Relevance</td>
    <td colspan="2">Diversity</td>
  </tr>
  <tr>
    <td>Rtrv</td>
    <td>Gene</td>
    <td>R-1</td>
    <td>R-2</td>
    <td>R-L</td>
    <td>Average</td>
    <td>Extrema</td>
    <td>Greedy</td>
    <td>Dist-1</td>
    <td>Dist-2</td>
  </tr>
  <tr>
    <td></td>
    <td>√ </td>
    <td>S2S+Attn</td>
    <td>35.36</td>
    <td>33.74</td>
    <td>18.16</td>
    <td>30.62</td>
    <td>0.341</td>
    <td>0.182</td>
    <td>0.368</td>
    <td>0.061</td>
    <td>0.081</td>
  </tr>
  <tr>
    <td></td>
    <td>√ </td>
    <td>CVAE</td>
    <td>43.65</td>
    <td>42.32</td>
    <td>21.83</td>
    <td>38.78</td>
    <td>0.358</td>
    <td>0.189</td>
    <td>0.373</td>
    <td>0.076</td>
    <td>0.201</td>
  </tr>
  <tr>
    <td></td>
    <td>√ </td>
    <td>UniLM</td>
    <td>49.31</td>
    <td>48.76</td>
    <td>31.89</td>
    <td>47.09</td>
    <td>0.383</td>
    <td>0.274</td>
    <td>0.389</td>
    <td>0.202</td>
    <td>0.364</td>
  </tr>
  <tr>
    <td>√ </td>
    <td></td>
    <td>Retrieval</td>
    <td>36.28</td>
    <td>30.19</td>
    <td>14.88</td>
    <td>28.36</td>
    <td>0.298</td>
    <td>0.164</td>
    <td>0.327</td>
    <td>0.131</td>
    <td>0.466</td>
  </tr>
  <tr>
    <td>√ </td>
    <td></td>
    <td>Retrieval+Rerank</td>
    <td>40.17</td>
    <td>36.49</td>
    <td>17.67</td>
    <td>35.44</td>
    <td>0.362</td>
    <td>0.211</td>
    <td>0.378</td>
    <td>0.137</td>
    <td>0.431</td>
  </tr>
  <tr>
    <td>√ </td>
    <td>√ </td>
    <td>Eidt</td>
    <td>47.65</td>
    <td>48.27</td>
    <td>29.81</td>
    <td>46.53</td>
    <td>0.378</td>
    <td>0.243</td>
    <td>0.366</td>
    <td>0.134</td>
    <td>0.189</td>
  </tr>
  <tr>
    <td>√ </td>
    <td>√ </td>
    <td>Ours</td>
    <td>55.1</td>
    <td>51.79</td>
    <td>32.07</td>
    <td>51.35</td>
    <td>0.394</td>
    <td>0.364</td>
    <td>0.391</td>
    <td>0.188</td>
    <td>0.297</td>
  </tr>
  <tr>
    <td rowspan="9">Weibo</td>
    <td colspan="2">Types</td>
    <td rowspan="2">Models</td>
    <td rowspan="2">BLEU</td>
    <td colspan="3">Rouge</td>
    <td colspan="3">Relevance</td>
    <td colspan="2">Diversity</td>
  </tr>
  <tr>
    <td>Rtrv</td>
    <td>Gene</td>
    <td>R-1</td>
    <td>R-2</td>
    <td>R-L</td>
    <td>Average</td>
    <td>Extrema</td>
    <td>Greedy</td>
    <td>Dist-1</td>
    <td>Dist-2</td>
  </tr>
  <tr>
    <td></td>
    <td>√ </td>
    <td>S2S+Attn</td>
    <td>37.21</td>
    <td>36.77</td>
    <td>20.14</td>
    <td>35.07</td>
    <td>0.341</td>
    <td>0.194</td>
    <td>0.366</td>
    <td>0.026</td>
    <td>0.084</td>
  </tr>
  <tr>
    <td></td>
    <td>√ </td>
    <td>CVAE</td>
    <td>44.74</td>
    <td>44.15</td>
    <td>23.12</td>
    <td>41.39</td>
    <td>0.358</td>
    <td>0.195</td>
    <td>0.378</td>
    <td>0.086</td>
    <td>0.142</td>
  </tr>
  <tr>
    <td></td>
    <td>√ </td>
    <td>UniLM</td>
    <td>50.06</td>
    <td>50.33</td>
    <td>32.19</td>
    <td>49.81</td>
    <td>0.388</td>
    <td>0.237</td>
    <td>0.387</td>
    <td>0.142</td>
    <td>0.342</td>
  </tr>
  <tr>
    <td>√ </td>
    <td></td>
    <td>Retrieval</td>
    <td>35.79</td>
    <td>32.41</td>
    <td>15.21</td>
    <td>28.03</td>
    <td>0.304</td>
    <td>0.162</td>
    <td>0.315</td>
    <td>0.111</td>
    <td>0.472</td>
  </tr>
  <tr>
    <td>√ </td>
    <td></td>
    <td>Retrieval+Rerank</td>
    <td>38.92</td>
    <td>38.29</td>
    <td>18.17</td>
    <td>35.14</td>
    <td>0.354</td>
    <td>0.201</td>
    <td>0.378</td>
    <td>0.167</td>
    <td>0.494</td>
  </tr>
  <tr>
    <td>√ </td>
    <td>√ </td>
    <td>Eidt</td>
    <td>50.64</td>
    <td>50.82</td>
    <td>26.71</td>
    <td>48.38</td>
    <td>0.396</td>
    <td>0.234</td>
    <td>0.387</td>
    <td>0.152</td>
    <td>0.158</td>
  </tr>
  <tr>
    <td>√ </td>
    <td>√ </td>
    <td>Ours</td>
    <td>56.3</td>
    <td>52.43</td>
    <td>29.07</td>
    <td>50.41</td>
    <td>0.412</td>
    <td>0.367</td>
    <td>0.411</td>
    <td>0.203</td>
    <td>0.314</td>
  </tr>
</tbody>
</table>

