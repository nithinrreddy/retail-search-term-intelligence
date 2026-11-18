# Labelling Rules: Search Term Relevance

Each search term is judged against one specific product, not in isolation.
The same query can be relevant for one product and unrelated for another.
Example: "hiking boots" is relevant for the TrailMax boot and unrelated for the Stride Pro running shoe.

## The three labels

**relevant**
The query names the advertised product type, brand or product name, and a shopper searching for it would be satisfied by this product.
- "running shoes" for the Stride Pro Running Shoe
- "northpeak stride pro" for the Stride Pro Running Shoe

**unrelated**
The query describes a different product type, or the shopper wants information rather than a product.
- "hiking boots" for the Stride Pro Running Shoe
- "smoothie recipes" for the ProBlend Blender (informational intent)

**uncertain**
The query is too broad to decide, or it names another brand for the same product type. A person should review these before any action is taken.
- "shoes" for the Stride Pro Running Shoe (too broad)
- "nike running shoes" for the Stride Pro Running Shoe (another brand, same product type)

## Quick decision steps

1. Does the query name this product type? If no, label it **unrelated**.
2. If yes, is it also naming another brand, or is it too broad to be sure? If yes, label it **uncertain**.
3. Otherwise, label it **relevant**.
4. If a query asks how to do something, or asks for facts, label it **unrelated**, even if it shares words with the product.

## Limitations

- The labels in this project were written by hand by the project author, using these rules. They have not been checked by a second annotator.
- All data is synthetic. Label patterns in synthetic data are cleaner than in real search reports, so model scores here will be higher than they would be on real data.
- Real labelling would need several annotators and an agreement score, such as Cohen's kappa, to show the rules are clear enough to apply consistently.
