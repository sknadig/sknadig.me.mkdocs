## Dialpad, Inc.

<p style="text-align:left;">
    Speech Recognition Engineer
    <span style="float:right;">
        Dec 2019 - Present
    </span>
</p>

- **Architect** the Dialpad next-gen ASR system with streaming end-to-end architectures
- **Lead the R&D** on streaming end-to-end ASR for conversational, telephony, and videoconferencing speech under low latency and multi accent scenarios
- **Benchmarked** various toolkits including Kaldi, K2, ESPnet, NeMo, and WeNet to architect the next-gen ASR system
- **Built and benchmarked** various end-to-end ASR architectures with CTC, Attention-based Encoder-Decoder (AED), Transducer, Transformer, and Conformer models with hybrid ASR models and external ASR services
- Developed interfaces for the **shallow fusion** of multi-level (sub-word and word) RNNLMs and n-gram LMs
- Developed methods to **bias the models** towards a list of keywords, resulting in an absolute WERR of **7%**
- Automated the **data preparation pipeline** for training ASR models, reducing the turnaround time for experiments and increasing productivity of the team
- Developed **pronunciation-assisted sub-word models** using fast-align, GIZA++, and Pynini, resulting in an absolute WERR of **3%** compared to BPE sub-words
- Post-training quantization of ASR models to achieve **50%** faster RTF and **75%** smaller models on disk
- Developed **performance monitoring** techniques for end-to-end ASR models based on RNN-AED and CTC confidence scores, and their efficacy in semi-supervised and self-supervised learning techniques
- Developed better **endpoint detection** for hybrid models and achieved **4%** relative WERR
- Developed a **web-app** for internal users to query production calls and visualize hypotheses using wavesurfer-js

### Open-source contributions:
- [MUCS 2021](https://www.youtube.com/watch?v=_ZGWXh3UMiI): MUltilingual and Code-Switching ASR Challenges for Low Resource Indian Languages
  - [Third Prize](https://navana-tech.github.io/MUCS2021/assets/img/winners/subtask1/3.PNG) in the challenge
  - Team contributions to multilingual and low-resource ASR for Indian Languages. Benchmarking and open-sourcing various end-to-end methods and studying effects of channel distortions on language identification
  - [Code available here](https://github.com/dialpad/mucs_2021_dialpad)

## Observe AI

<p style="text-align:left;">
    Machine Learning Intern - ASR
    <span style="float:right;">
        May 2019 - Aug 2019
    </span>
</p>

- Developed a feature extraction pipeline using `tf.signal` and `tf.data`
- Implemented different keyword-spotting (KWS) papers - Deep-KWS, CTC KWS
- Developed methods to convert a custom PyTorch model to TensorFlow
- Deployed the KWS model using TensorFlow serving with an **RTF of 0.05** on GPU 

## IIIT-Bangalore

<p style="text-align:left;">
    Research Scholar
    <span style="float:right;">
        Jan 2017 - Dec 2019
    </span>
</p>

- Developed end-to-end methods for multilingual and code-switching scenarios in Indian Languages
- Developed joint ASR and KWS systems using joint phoneme-grapheme recognition
- Developed a more accurate and faster training method by jointly training alignment and ASR model
- Mentored MTech and iMTech students in their projects and thesis work and delivered various tutorials and talks around ASR
- Developed a remote hardware laboratory to study control systems using embedded programming and web technologies
- Developed HCI visualizations for a humanoid using Unity and C#

I was also involved in different labs and activities including:
- [E-Health Research Center (EHRC)](https://ehrc.iiitb.ac.in/)
  - Developed rehabilitation robotics applications
- [Machine Intelligence and Robotics Center (MINRO)](https://minro.org/)
  - Developed multi-lingual applications of Speech and Language Technologies in the domain of e-governance.
- [Intel AI Academy Student Ambassador](https://software.intel.com/en-us/ai/ambassadors).
  - Built a small-footprint ASR application ([keyword spotting](https://devmesh.intel.com/projects/end-to-end-asr-with-intel-ncs-262635), wake-word detection) on the "edge" using Intel's Neural Compute Stick 2 (NCS2) and OpenVINO.
- Graduate Teaching Assistant
  - Deep Learning for Automatic Speech Recognition
  - Automatic Speech Recognition
  - Introduction to Robotics

### Invited Talks
- <p style="text-align:left;"> <a href="https://youtu.be/J5TOt_bKVzI">IIIT-B Samvaad talk</a> <span style="float:right;">Bengaluru, IN | Dec 2020</span> </p>
    - Multi-task learning in end-to-end attention-based automatic speech recognition (MS Thesis)
    - Open challenges in multi-task learning for ASR

- <p style="text-align:left;"><a href="#invited-talks">IIIT-B Guest Lecture Series - Deep Learning for ASR</a><span style="float:right;">Bengaluru, IN | Sep 2020 - Dec 2020</span> </p>
    - Discussions on RNN-CTC, RNN-AED, RNN-T, and Transformer models for ASR
    - Discussions on model quantization and weight sparsity in RNN-T models for low computational resource and latency constraints
    - Unpacking and analyzing the Pixel Recorder app to showcase how tflite models are packed with custom TensorFlow ops

- <p style="text-align:left;"> <a href="https://github.com/sknadig/BMSCE_workshop">Artificial Intelligence : A Way Forward</a> <span style="float:right;">Bengaluru, IN | Sep 2019</span> </p>
    - Faculty development program at Dayananda Sagar College of Arts, Science and Commerce, Bangalore
    - Discussions on the use of AI in speech and language technology

- <p style="text-align:left;"> <a href="https://github.com/sknadig/TCS_TL_e2e_ASR">TCS Think Labs</a> <span style="float:right;">Bengaluru, IN | Feb 2019</span> </p>
    - Motivation and introduction to end-to-end ASR
    - Discussion on the topics of RNN, CTC, Attention and LM fusion

- <p style="text-align:left;"> <a href="https://github.com/sknadig/attention_presentation/blob/master/Final.pdf">IIIT-B AI Reading Group</a> <span style="float:right;">Bengaluru, IN | Nov 2018</span> </p>
    - Discussions on various attention models in end-to-end ASR
    - Semi-supervised learning with end-to-end ASR models

- <p style="text-align:left;"> <a href="https://github.com/sknadig/BMSCE_workshop">BMSCE AI Workshop</a> <span style="float:right;">Bengaluru, IN | Sep 2018</span> </p>
    - Artificial Intelligence and Deep Neural Networks Workshop for undergraduate students at BMS College of Engineering, Bangalore
    - Code examples and tutorials in TensorFlow Keras

## Sonus Networks

<p style="text-align:left;">
    SVT Engineer
    <span style="float:right;">
        Aug 2015 - Jan 2017
    </span>
</p>

- Worked as a part of Sustaining SVT on Real-Time communication products Sonus Insight (EMS) and SBC
- Developed automated test frameworks in Python, Perl, Linux, and Java
- Worked with CentOS, Red Hat Enterprise Linux, and Solaris to develop and test the products
- Developed tools that reduced team effort from many hours to a couple of minutes

<style>
.alignleft {
	float: left;
}
.alignright {
	float: right;
}
</style>