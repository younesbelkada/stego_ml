NUM_OF_ARTICLES = 200

def gen_fake_text(mod, tok, start_text, start_secret, secret_text):
  ranks = get_ranks(mod, tok, start_secret, secret_text)
  cover_text = generate_cover_text(mod, tok, start_text, ranks)

  return cover_text

def get_news(corpus_size=NUM_OF_ARTICLES, secret_text_length=5, mod, tok, start_secret, secret_text):

    news_text = brown.sents(categories=['news'])[:NUM_OF_ARTICLES]
    fake_news = []
    true_news = []

    for sentence in news_text:
        if len(sentence) <= secret_text_length:
            continue

        sentence_true = ' '.join(sentence)

        sentence_true = sentence_true.lower()

        true_news.append(sentence_true)

        sentence_fake = ' '.join(sentence[:-secret_text_length])
        print(sentence_fake)
        sentence_fake = gen_fake_text(mod, tok, sentence_fake, start_secret, secret_text)

        fake_news.append(sentence_fake)
    
    return true_news, fake_news

def prepare_news(true_news, fake_news):
    punctuation_dict = {}
    for x in string.punctuation:
        punctuation_dict[ord(x)] = None

    print(true_news[1].translate(punctuation_dict))

    for i in range(len(true_news)):
        true_news[i] = true_news[i].translate(punctuation_dict)

    for i in range(len(fake_news)):
        fake_news[i] = fake_news[i].translate(punctuation_dict)

    # s.translate({ord(c): None for c in string.whitespace})
    
    for i in range(len(true_news)):
        true_news[i] = true_news[i].lower()
        
    for i in range(len(true_news)):
        true_news[i] = true_news[i].strip()

    for i in range(len(fake_news)):
        fake_news[i] = fake_news[i].strip()

def prepare_df_for_fasttext():
    pass
