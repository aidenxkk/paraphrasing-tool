This project aims to design a paraphrasing tool for protecting user privacy by transforming English text inputs into paraphrased outputs while preserving the original meaning. It uses sentence embeddings as encoders. The tool also tries to improve zero-shot learning(ZSL) performance by using Semantic Autoencoder(SAE).  

Project Components
Target: Design a paraphrasing tool that safeguards user privacy by rephrasing English text while preserving its semantic meaning.
Input: A list of sentences.
Output: A list of paraphrased sentences.
Tools:
vec2Text pretrained model.
Semantic Autoencoder (SAE).
Challenges
Integrating the vec2Text pretrained model correctly.
Matching the encoding method with the input format expected by the pretrained model.
Ensuring robust performance across diverse inputs.
Steps: encoder the sentence input to semantic embeddings - feed embeddings to model - decode the embeddings to sentence - repeat enough time to train the model for a specific similarity threshold. 
