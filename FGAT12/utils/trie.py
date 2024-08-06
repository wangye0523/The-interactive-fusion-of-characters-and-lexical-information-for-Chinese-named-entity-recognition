import collections
class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = collections.defaultdict(TrieNode) # 一个dict的子类，可以使创建的字典具有默认值
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word): #将词语插入树中，每个结点是个字。
        
        current = self.root
        for letter in word:
            current = current.children[letter]  #如果letter不存在于键中，那么返回默认值，这里的默认值是TrieNode key=letter,value=TrieNode
        current.is_word = True

    def search(self, word): #查找词语
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False
        return current.is_word

    def startsWith(self, prefix): #查找字符串开头部分的子串（prefix是某个词的开头则返回True）
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True


    def enumerateMatch(self, word, space="_", backward=False):  #space=‘’
        matched = []

        while len(word) > 0:
            if self.search(word):
                matched.append(space.join(word[:]))
            del word[-1]
        return matched

