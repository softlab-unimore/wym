# WYM

An Explainable classification systems generate predictions along with a weight for each pair of term (decision unit) in the input record measuring its contribution to the prediction.
In the entity matching (EM) scenario, inputs are pairs of entity descriptions and the resulting explanations can be difficult to understand for the users.
They can be very long and assign different impacts to similar terms located in different descriptions.

#### We introduce the concept of decision units, basic information units formed either by pairs of (similar) terms, each one belonging to a different entity description, or unique terms, existing in one of the descriptions only.

#### Decision units form a new feature space, able to represent, in a compact and meaningful way, pairs of entity descriptions.

#### An explainable model trained on such features generates effective explanations customized for EM datasets. 

In this paper, we propose this idea via a three-component architecture template, which consists of a decision unit generator, a decision unit scorer, and an explainable matcher.

### WYM Logic Architecture
![WYM Logic Architecture`](LogicArchitecture.svg)

#### We introduce WYM (Why do You Match?), an implementation of the architecture oriented to textual EM databases.
The experiments show that our approach has accuracy comparable to other state-of-the-art Deep Learning based EM models, but, differently from them, its prediction are highly interpretable.

[//]: # (### WYM full architecture)

[//]: # (![WYM flow`]&#40;Architecture.svg&#41;)


### Quick Start: WYM in 30 seconds

```sh
git clone https://github.com/softlab-unimore/wym.git
pip install -e wym
```
Import

```python
from wym.wym import Wym
```
Initialize it.

```python
wym = Wym(df=train_df, exclude_attrs=exclude_attrs)
```

You can use the `fit` - `predict` interface as an sklearn learner.
```python
X, y = train_df[wym.columns_to_use], train_df['label']
wym.fit(X, y, X_valid, y_valid)
match_score = wym.predict(X_test)
```
You can also plot decision unit explanations.
```python
match_score, data_dict, word_pairs, emb_pairs, features, word_relevance = wym.predict(X_test, return_data=True)
el_exp = word_relevance.query(f'id == 0')
wym.plot_token_contribution(el_exp)
```

### Working example

Get started with WYM!

[Here](https://github.com/softlab-unimore/wym/blob/main/quick_start_wym.ipynb)
you can find a working notebook where WYM run over a sample dataset.




