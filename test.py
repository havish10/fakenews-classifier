import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('./fake_or_real_news.csv')
df_test = pd.read_csv('./test_true.csv')

input_y = df_test.label
df_test = df_test.drop('label', axis=1)
input = df_test['text']

print(type(input))
print(input)

y = df.label
df = df.drop('label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

# Test Input
count_vectorizer_input = CountVectorizer(stop_words='english')
count_train = count_vectorizer_input.fit_transform(X_train)
count_input = count_vectorizer_input.transform(["CHRISTCHURCH, New Zealand — When the prime minister announced plans to ban semiautomatic rifles following Friday’s mass shootings, it seemed to be the bold response that many New Zealanders wanted — until the country’s attorney general backpedaled almost immediately and said that might not be the government’s final decision. Even after a massacre that left 50 people dead, the fight over guns and safety will be a fraught one for politicians in peaceful New Zealand, just as it is in the United States. But there is a crucial difference between the two countries that is already apparent: While Washington struggles to take action even as such shootings become more routine, New Zealand’s government is immediately diving into a detailed discussion of further legislative checks on guns. The outright prohibition of semiautomatic weapons proposed by Prime Minister Jacinda Ardern may no longer be on the near-term table. But she’s made clear that lawmakers will look at a range of options, from gun buybacks to restrictions on magazines for semiautomatic rifles. “New Zealand has to have this debate,” said Alexander Gillespie, a law professor at the University of Waikato, who once predicted that the country’s approach to firearms would lead to more mass shootings. “This is a place where your car has to be registered, your dog has to be registered. But your gun doesn’t.” New Zealand’s relationship to firearms amounts to a sliding scale of restrictions. Gun owners need a license, but the most commonly used guns, like hunting rifles, are never registered and can be easily bought and sold in large quantities. Handguns and semiautomatic weapons are more closely tracked, requiring a permit for each purchase and a separate license — making it harder but not impossible to amass an arsenal. The mix of freedom and regulation reflects the country’s frontier history, according to experts, who note that New Zealand’s link to weaponry bears a resemblance to the United States, Australia and Canada, but with a few important distinctions."])

# load the model from disk
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(count_input)
print(result)