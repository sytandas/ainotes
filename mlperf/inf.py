# how to do mlperf on tinygrad stable diffusion!!!
# findings: should be written from scratch after understanding the codebase and goals.
# old commit are unclear  > need of research 
# 
# something about scheduling and handling the code from kernal level to output
# the type of document he came of by reverse engineering that has to be learn for deep and practical understanding
# there are still complex ideas in tandom with lazydata, realize 
# realize is what data should be allocated memory in the GPU contains 6 lazydata - MUL1, COPY2, EXT8, CONST7, MUL10 so on need to check
# realizes using directly to generate schedule items
# allbufs just to use stop creating same tree again
# lazy buffer is casting data to chneel to process at my language and iterate through _scheduler_lb
