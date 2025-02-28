# Paper Replication: Do Llamas Work in English? On the Latent Language of Multilingual Transformers

This is a replication of the paper [Do Llamas Work in English? On the Latent Language of Multilingual Transformers](https://arxiv.org/abs/2402.10588), as well as additional experiments expanding the work of the paper, on which we get results similar to the paper' results.

This project was done in collaboration with Anthony Duong. It was started as the capstone project of the ARENA bootcamp and continued after the end of the bootcamp.

## Experiments

The approach is to do [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) on a model run on text not in English. We observe that on later layers, the model "thinks" the tokens in the language the prompt is in, but in middle layers, it also "thinks" in English and Chinese a significant amount, ofen more than in the language of the prompt. (Note that it is unclear wether it makes sense to say the model "thinks" something based on looking at logit lens tokens.)

The original paper looks at a toy dataset of translation and repetition tasks.
Our approach is different: we take real texts and then translate all tokens which aren't stop words, punctuation, etc to different langauges.
We measure the probabilities logit lens assigns to each token's translations.

We run the experiments on Qwen, which is a Chinese model.
We think it is interesting to see whether Chinese models also think in English, and the original paper only used Western models.
(Note that, as the original paper says, it is unclear how good a proxy the method is for "what language does the model think in?" See the paper for a more thorough discussion)

## Results

We observe that, at late layers, logit lens assigns high probabilites to tokens in the target language.
On middle layers, the model often assigns high probabilities to the same tokens translated to Enlgish and Chinese.
This suggests the model might "think" in English or Chinese, and then translate its output to the language the prompt is in.

TO DO: Run this on bigger samples. The error bars are too wide.

TO DO: plots
![plot](figures/1.png)
![plot](figures/2.png)
![plot](figures/3.png)
![plot](figures/4.png)
