# Human-AI Collaborative Essay Scoring: A Dual-Process Framework with LLMs

<!-- This repository contains the codes for the paper [*Human-AI Collaborative Essay Scoring: A Dual-Process Framework with LLMs*](https://arxiv.org/abs/2401.06431).  -->

This repository contains the codes for the paper *Human-AI Collaborative Essay Scoring: A Dual-Process Framework with LLMs*. Dataset and other resources will be released after the anonymous period. 


# Overview


In this study, we explore the potential of proprietary and open-source LLMs such as **GPT-3.5, GPT-4, and LLaMA3** for **Automated Essay Scoring (AES)**. We conducted extensive experiments with public ASAP dataset as well as a private collection of student essays to assess the zero-shot and few-shot performance of these models. Additionally, we enhanced their effectiveness through supervised fine-tuning (SFT). Drawing inspiration from the **dual-process theory**, we developed an AES system based on LLaMA3 that matches the grading accuracy and feedback quality of fine-tuned LLaMA3. 

Our **human-LLM co-grading experiment** further revealed that this system significantly improves the performance and efficiency of both novice and expert graders, offering valuable insights into the educational impacts and potential for effective human-AI collaboration.

![example](img/first_page_new.png)

# Contributions

- **LLMs in AES**: We pioneer the exploration of LLMs' capabilities as AES systems, especially in complex scenarios featuring tailored grading criteria. Leveraging dual-process theory, our novel AES framework demonstrates **remarkable accuracy, efficiency, and explainability**.

- **Dataset**: We introduce an extensive essay-scoring dataset, which includes **13,372 essays** written by Chinese high school students. These essays are evaluated with multi-dimensional scores by expert educators. 

- **Human-AI Collaboration**: Our findings from the human-LLM co-grading task highlight **the potential of LLM-generated feedback to elevate the proficiency of individuals with limited domain expertise to a level akin to that of experts**. Additionally, it enhances the efficiency and robustness of human graders by integrating model confidence scores and explanations. These insights set the stage for future investigation into human-AI collaboration and AI-assisted learning within educational contexts. 


