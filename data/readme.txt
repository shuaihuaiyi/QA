1.Task Specification
Given a question and it's corresponding document, your system should select one or more sentences as answers from the document.

2.Data Format
1).An example:
芝加哥国际电影节最佳女主角是谁？ \t 芝加哥国际电影节(Chicago International Film Festival)是北美历史最久的评奖电影节。 \t 0
芝加哥国际电影节最佳女主角是谁？ \t 最佳女主角奖：余男《图雅的婚事》（中国） \t 1
芝加哥国际电影节最佳女主角是谁？ \t 芝加哥电影节每年10月举办，自1965年第一届电影节至今，芝加哥国际电影节已成为世界知名的年度电影盛会。 \t 0
芝加哥国际电影节最佳女主角是谁？ \t 我国著名电影导演张艺谋的《菊豆》曾于1990年首获该电影节最高奖--金雨果奖。 \t 0
芝加哥国际电影节最佳女主角是谁？ \t 芝加哥电影节组织是由电影制作人、形象艺术家麦克尔.库泽于1964年发起成立的。 \t 0
芝加哥国际电影节最佳女主角是谁？ \t 其宗旨为：通过电影和录像艺术手段加强不同文化背景人民之间的理解和沟通。 \t 0
芝加哥国际电影节最佳女主角是谁？ \t 评委会特别大奖：《图雅的婚事》（中国） \t 0
芝加哥国际电影节最佳女主角是谁？ \t 2002年，来自30多个国家和地区的90多部故事片、40多部短片参加了电影节，吸引了世界各地的6万多观众。 \t 0

2).Explanation
A question (the 1st column), question’s corresponding document sentences (the 2nd column), and their answer annotations (the 3rd column) are provided. 
If a document sentence is the correct answer of the question, its annotation will be 1, otherwise its annotation will be 0. 
The three columns will be separated by the symbol ‘\t’.
All the dataset file are encoded in UTF-8.

3.Data Statistics
dataset             # of unique questions
train set           7895
development set     878
test set            5997

4.Evaluation Metrics
MRR, MAP, and ACC@1.


5. 输出文件格式

每行只包括一个实数，表示改行问句和答案候选句之间的关系。


形如：

0.1534
2.7762
0.0097
15.2345
.
.
.
.
.

评价工具会按照数值大小，来对每个问句的结果进行排序。
