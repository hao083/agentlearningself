#算法思想
'''
第一步：用户输入一个问题，系统将问题传递给ELIZA算法。
第二步：ELIZA算法将用户输入的问题进行分析，根据预定好的优先级进行匹配，从而确定关键字
第三步：根据预设好的锚点（注意，锚点不是关键字，可以是关键字）将用户问题进行分解，分为多组%1，%2......%n
第四步：从规则库中进行匹配，找到最合适的规则，并将用户问题中的关键字替换到规则中，形成一个暂时的回应
第五步：进行人称转换，第一人称变为第二人称，例如我变成你，我的变成你的，这样会显得回答更亲切
第六步：向用户输出回应
'''


#算法实现

#构建规则库，开发者进行手工书写
import re
import random

# 定义规则库：模式(正则表达式) -> 响应模板列表
rules = {
    r'I need (.*)': [
        "Why do you need {0}?",
        "Would it really help you to get {0}?",
        "Are you sure you need {0}?"
    ],
    r'Why don\'t you (.*)\?': [
        "Do you really think I don't {0}?",
        "Perhaps eventually I will {0}.",
        "Do you really want me to {0}?"
    ],
    r'Why can\'t I (.*)\?': [
        "Do you think you should be able to {0}?",
        "If you could {0}, what would you do?",
        "I don't know -- why can't you {0}?"
    ],
    r'I am (.*)': [
        "Did you come to me because you are {0}?",
        "How long have you been {0}?",
        "How do you feel about being {0}?"
    ],
    #增加两条洗的规则
    r'I feel (.*)': [
        "Why do you feel {0}?",
        "How long have you been feeling {0}?",
        "What makes you feel {0}?"
    ],
    r'.*school.*':[
        "Could you tell me more about your school?",
        "What do you like about your school?",
        "Are there any interesting places near your school?"
    ],
    r'.*basketball.*':[
        "Do you like watching the NBA?",
        "What is your favorite basketball team?",
        "How did you get into basketball?"
    ],
    r'.*mother.*': [
        "Tell me more about your mother.",
        "What was your relationship with your mother like?",
        "How do you feel about your mother?"
    ],
    r'father': [
        "Tell me more about your father.",
        "How did your father make you feel?",
        "What has your father taught you?"
    ],

    #采用第二种记忆，要删去通用匹配规则
    r'.*': [
            "Please tell me more.",
            "Let's change focus a bit... Tell me about your family.",
            "Can you elaborate on that?"
    ]
    
}

# 定义代词转换规则
pronoun_swap = {
    "i": "you", "you": "i", "me": "you", "my": "your",
    "am": "are", "are": "am", "was": "were", "i'd": "you would",
    "i've": "you have", "i'll": "you will", "yours": "mine",
    "mine": "yours"
}

def swap_pronouns(phrase):
    """
    对输入短语中的代词进行第一/第二人称转换
    """
    words = phrase.lower().split()
    swapped_words = [pronoun_swap.get(word, word) for word in words]
    return " ".join(swapped_words)

'''
#无记忆功能的回应
def respond(user_input):
    """
    根据规则库生成响应
    """
    for pattern, responses in rules.items():
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            # 捕获匹配到的部分
            captured_group = match.group(1) if match.groups() else ''
            # 进行代词转换
            swapped_group = swap_pronouns(captured_group)
            # 从模板中随机选择一个并格式化
            response = random.choice(responses).format(swapped_group)
            return response
    # 如果没有匹配任何特定规则，使用最后的通配符规则
    return random.choice(rules[r'.*'])
'''
'''
##增加记忆功能
#------------------方案一，尽量不重复回应--------------------
last_pattern = None    #用于记录上一次命中的规则模式

def respond(user_input):
    global last_pattern  # 声明使用其修改全局变量
    items=list(rules.items())

    for pattern,responses in items:
        match = re.search(pattern,user_input,re.IGNORECASE)
        if match:
            captured = match.group(1) if match.groups() else''
            swapped = swap_pronouns(captured)

            #检查是否和上次是同一条规则，如果是，那就更换
            if pattern == last_pattern and len(responses)>1:#匹配到相同的规则
                #过滤上一次使用的response，避免重复的情况出现
                last_response=getattr(respond,"_last_response",None)
                choices = [r for r in responses if r !=last_response]
                template = random.choice(choices if choices else responses)
            else:#不相等，则从匹配到的规则随机挑选
                template = random.choice(responses)

            #更新记忆
            last_pattern=pattern

            #格式化模板，也就是拼接操作
            try:
                response = template.format(swapped)
            except (IndexError, KeyError):
                #模板含有字面花括号等异常情况，返回原模版
                response = template

            #更新记忆，记录本轮输出
            respond._last_response = response
            return response

    #不匹配，最后兜底
    return random.choice(rules[r'.*'])
'''

#增加记忆功能
#-------------------方案二，对话历史＋话题stack--------------------
#采用标外挂一个“记忆容器”,用来存放对话历史+话题stack
#能够记住用户通过的话题，当触发兜底时回访，而不是干巴巴的敷衍

class ElizaMemory:
    '''
    记忆容器：
    history:存放完整对话记录
    topics:提及过的话题（list当作stack去使用，采用后进先出）
    last_pattern:上一轮命中的规则
    该算法自带“限长”，从而防止无限增长，这是最早的上下文窗口
    '''
    #初始化
    def __init__(self,max_turns=20,max_topics=10):
        self.history=[]#对话历史
        self.topics=[]#话题stack
        self.last_pattern=None#上一轮命中的规则
        self.max_turns=max_turns#对话历史最大长度
        self.max_topics=max_topics#话题stack最大长度
    
    #增加一轮对话，一轮是指Eliza+user
    def add_turn(self,user_text,response,matched_pattern):
        '''记录一轮对话，用户机器隔一条'''
        self.history.append(("user",user_text))#向对话记录数组中增加用户的话
        self.history.append(("Eliza",response))
        self.last_pattern = matched_pattern 
        #限制长度
        if len(self.history)>self.max_turns*2:#因为机器和任务各占一条，所以乘以2
            self.history=self.history[-self.max_turns*2:]#代替最旧的对话记录

    #增加最新的话题到话题ayy中，超出长度就去除最旧的话题
    def remember_topic(self,keyword):
        '''记住一个话题（去重+限长）'''
        if (keyword) and (keyword not in self.topics):
            self.topics.append(keyword)#not esist,and append keyword
            #judge the length of self.topics ayy yes/no over the self.max_topics
            if len(self.topics)>self.max_topics:#over the length
                self.topics = self.topics[-self.max_topics:]#保留较新的self.amx_topics数量的话题，旧的丢弃

    #话题出栈，弹出最新的话题
    def pop_topic(self):
        '''查看并且弹出'''
        return self.topics.pop() if self.topics else None

    def last_topic(self):
        '''只查看，不弹出'''
        return self.topics[-1] if self.topics else None

#单用户场景
em=ElizaMemory()
'''
memory=new ElizaMemory
memory._init_()
这是错误写法！！！m
'''


def respond(user_input):
    '''
    生成回应函数，相较于原版，增加了“话题记录”和“兜底回访”
    '''
    #按插入顺序进行遍历规则
    for pattern,responses in rules.items():
        match=re.search(pattern,user_input,re.IGNORECASE)
        if (match):
            captured=match.group(1) if match.groups() else''
            swapped=swap_pronouns(captured)
            #捕获的内容写入记忆中
            if(captured):
                em.remember_topic(captured)
            #识别话题，写入话题
            for kw in ["mother","father","basketball","school"]:
                if kw in user_input.lower():
                    em.remember_topic(kw)
            #选取模板并拼接格式化
            template = random.choice(responses)
            try:
                #读取记忆
                if "{0}" in template and not captured and em.last_topic():
                    response=template.format(em.last_topic())
                else:
                    response=template.format(swapped)
            except (IndexError, KeyError):
                response = template

              # ---- ★ 记忆写入③：记录本轮对话 ----
            em.add_turn(user_input, response, pattern)
            return response

    # ---- ★ 兜底时利用记忆：回访最近话题，而非直接敷衍 ----
    if em.last_topic():
        return f"Earlier you mentioned {em.last_topic()}. Tell me more about that."
    return random.choice(rules[r'.*'])


# 主聊天循环
if __name__ == '__main__':
    print("Therapist: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Therapist: Goodbye. It was nice talking to you.")
            break
        response = respond(user_input)
        print(f"Therapist: {response}")