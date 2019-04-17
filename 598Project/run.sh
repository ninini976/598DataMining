python f.py
rm ../GCN/gcn/gcn/data/ind.BuzzFeedSample*
cp ind.BuzzFeedSample* ../GCN/gcn/gcn/data
python ../GCN/gcn/gcn/train.py --dataset BuzzFeedSample182