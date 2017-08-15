import praw
import pickle

pick = True
global_count = 0

def dive(comment, array):
    tree = []
    array.append((comment, tree))
    for reply in comment.replies:
        dive(reply, tree)

def print_tree(nodes, depth):
    global global_count
    for node in nodes:
        print(node[0].body)
        global_count += 1
        print_tree(node[1], depth+1)

if pick:
    base_tree = pickle.load(open('top1000.p', 'rb'))
else:
    reddit = praw.Reddit(client_id='YOUR_ID_HERE', client_secret='YOUR_SECRET_HERE', user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.96 Safari/537.36')
    base_tree = []
    count = 0

    for submission in reddit.subreddit('totallynotrobots').top(limit=1000):
        print(count)
        count += 1
        tree = []
        base_tree.append((submission, tree))
        submission.comments.replace_more(limit=0)
        for comment in submission.comments:
            dive(comment, tree)

    pickle.dump(base_tree, open('top1000.p', 'wb'))
