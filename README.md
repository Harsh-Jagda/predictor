# Rental Price Prediction and Error Analysis

This repository is the culmination of an experiment that began with a simple question: how predictable are rental prices, really? At first glance, rents appear to be a straightforward reflection of area, location, and property type. Yet the deeper I dug, the clearer it became that rental markets are stubbornly complex, driven not only by measurable features but also by the noise of human decisions, timing, and market quirks. What follows is the story of how I tried to untangle this.

## Framing the Problem

The starting point was deceptively simple: predict monthly rents of residential properties given their attributes. The goal was not only to build a model that could estimate rents but also to capture the errors in those predictions and turn them into something interpretable. For me, the error itself became just as interesting as the prediction. Why do models miss in the way they do? Can I cluster those misses, map them, and learn from them?

With that motivation, the project quickly evolved from a predictive exercise into a pipeline that could generate insights about pricing trends, mispricing, and the overall behavior of the rental market.

## The Data

The dataset was structured with the essentials:

- **Address, Location, City**: context for where the property sits geographically
- **Rent, Predicted Rent, Error, Error Percent**: the ground truth vs the model’s estimate, plus absolute and relative error
- **Beds, Baths, Type, Area in sqft, Furnishing, Purpose**: features that describe the property
- **Geo coordinates and cluster labels**: a way to group properties spatially
- **Derived features**: such as rent per square foot, error categories, and price status labels

Each column carried a purpose. Some acted as direct explanatory features. Others were engineered after predictions to help make sense of model blind spots. For example, **Error_Percent** was carefully normalized so that it always represented a fraction (0.05 meaning 5%), avoiding misleading infinite percentages that appear when dividing by very small rents.

## Modeling Choices

At the modeling stage, my emphasis was not on using the most exotic algorithm available but rather on building a foundation I could trust. I experimented with a range of models, beginning with straightforward baselines and gradually scaling toward more complex approaches. The core thread running through all of them was interpretability. A prediction on its own is inert; it only becomes useful when I can explain why it was made and where it fails.

I treated the process like an investigation. Linear models revealed which features carried the strongest weight in a transparent way. Tree-based models, especially gradient boosting, added the ability to capture nonlinear interactions that dominate real estate markets, such as the compounding effect of location and size. Alongside this, I examined feature importances, tested regularization to counter overfitting, and tuned hyperparameters with a careful balance of accuracy and generalization.

## Error Analysis

One of the key design decisions was to focus explicitly on the errors. Every prediction was paired with its absolute error, relative error, and categorical interpretation of whether the model overestimated or underestimated. This allowed me to do more than just track accuracy. I could ask:

- Where is the model consistently optimistic?
- Where is it systematically pessimistic?
- Which clusters of the city show higher mispricing rates?

The **MAPE** (Mean Absolute Percentage Error) became a central metric, but I treated it with care. Because rents can be low in some parts of the data, I had to guard against division by near-zero values that would inflate percentage errors. Cleaning this step meant that the MAPE I reported was a truer reflection of model behavior.

## Visualization

Once the structure was in place, I leaned heavily into visualization. The idea was not only to validate the model but to tell a story through maps, scatter plots, and error distributions. Some highlights include:

- **Scatter plots of actual vs predicted rents** that immediately revealed underestimation of higher-end properties
- **Error distributions** that showed whether the model leaned toward systematic bias or random noise
- **Geographic clustering** where I plotted the mispredictions directly on a map, revealing pockets of overpricing or underpricing by neighborhood
- **Feature-level breakdowns** such as rent per square foot across furnishing types or property categories

Together, these visualizations allowed me to shift the perspective from individual predictions to market-level behavior.

## The Why

The real motivation was never just about building a rent predictor. It was about understanding why predictive models fail in the messy reality of human-driven markets. Every error is a signal: maybe the dataset is missing a feature that really matters, like proximity to a new transit line or the prestige of a neighborhood. Maybe the data captured a market shock or an idiosyncratic landlord decision. By analyzing errors instead of hiding them, I turned the project into an exploration of the limitations of data science in the real world.

## What Comes Next

This project, as it stands, provides both predictions and an analytical framework to interpret them. It is a snapshot of how a model views the rental market and how its vision warps in certain areas. The natural next steps are to expand the feature set, refine the clustering, and experiment with more robust ensemble methods. But equally important is the continued commitment to transparency: not just asking how well the model performs but also where and why it stumbles.

In the end, this work is less about a perfect predictor and more about the dialogue between data, model, and error. It shows that the distance between predicted rent and actual rent is not just a mistake to minimize but a lens through which the deeper forces of the rental market can be observed.
