# WYM

We introduce an architecture template for addressing interpretable entity matching composed of three main components. The first one computes decision units, which are the basic information unit on which we base the approach. The decision units can be formed by pairs of terms, each one belonging to a different entity description, or unique terms, existing in one of the descriptions only.
Then, the second component provides a relevance score for each decision unit. Finally, the third component provides the matching prediction along with impact scores associated with the decision units, which give a measure for  (i.e. an explanation of) their importance in the matching decision.

We implement this architecture template with *WYM* (Why do You Match?), an approach oriented to textual databases, where word embeddings and a greedy implementation of an assignment algorithm generates the decision units, a regression Neural Network model computes the relevance scores, a classification model, trained on a dataset created from the relevance scores, predict the entity matching and through its coefficients provides the impact scores constituting the explanation.
The experiments show that our approach has accuracy comparable to other state-of-the-art Deep Learning based EM models, but, differently from them, the impact scores make *WYM*   highly interpretable.

![WYM flow`](Architecture.svg)