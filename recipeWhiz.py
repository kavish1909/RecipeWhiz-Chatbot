#The code in this file is referred from https://github.com/jezhiggins/eliza.py

#importing necessary libraries
import string
import re
import random

class RecipeWhiz:
  def __init__(self):
    self.keys = list(map(lambda x: re.compile(x[0], re.IGNORECASE), gPats))
    self.values = list(map(lambda x: x[1], gPats))

  def translate(self, text, vocabulary):
    words = text.lower().split()
    keys = vocabulary.keys();
    for i in range(0, len(words)):
      if words[i] in keys:
        words[i] = vocabulary[words[i]]
    return ' '.join(words)


  def respond(self, text):
    # find a match among keys
    for i in range(0, len(self.keys)):
      match = self.keys[i].match(text)
      if match:
        # found a match ... stuff with corresponding value
        # chosen randomly from among the available options
        resp = random.choice(self.values[i])
        # we've got a response... stuff in reflected text where indicated
        pos = resp.find('%')
        while pos > -1:
          num = int(resp[pos+1:pos+2])
          resp = resp[:pos] + \
            self.translate(match.group(num), gReflections) + \
            resp[pos+2:]
          pos = resp.find('%')
        # fix munged punctuation at the end
        if resp[-2:] == '?.': resp = resp[:-2] + '.'
        if resp[-2:] == '??': resp = resp[:-2] + '?'
        return resp
    return None
    
#dictionary to replace nouns
gReflections = {
  "am"   : "are",
  "was"  : "were",
  "i"    : "you",
  "i'd"  : "you would",
  "i've"  : "you have",
  "i'll"  : "you will",
  "my"  : "your",
  "are"  : "am",
  "you've": "I have",
  "you'll": "I will",
  "your"  : "my",
  "yours"  : "mine",
  "you"  : "me",
  "me"  : "you"
}

#defining patterns for identifying intent of user text and provide appropriate reponses
gPats = [
    [r'quit',
    [ "Thank you.  Have a good day!"]],
    [r'Can you',
    [ "Please provide me with more information."]],
    [r'.*',
    [  "Sorry I don't understand you. Could you please provide information related to dishes so that I can help you."]]
]
