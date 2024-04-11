import random

import gensim
import numpy as np
import torch
from gensim.models import KeyedVectors
from torch.utils.data import Dataset

from data_augmentation import Crop, Mask, Reorder, Substitute, Insert, Random, CombinatorialEnumerate
from utils import neg_sample, nCr
import copy


class RecWithContrastiveLearningDataset(Dataset):
    def __init__(self, args, user_seq, rating_seq,test_neg_items=None, data_type='train',
                similarity_model_type='offline', all = False):
        self.args = args
        self.user_seq = user_seq
        self.ratings = rating_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length
        self.all = all
        self.model = gensim.models.Word2Vec.load(args.data_dir + 'Word2vec_'+args.data_name + '.bin')

        # currently apply one transform, will extend to multiples
        # it takes one sequence of items as input, and apply augmentation operation to get another sequence
        self.similarity_model = args.offline_similarity_model
        print("Similarity Model Type:", similarity_model_type)
        self.augmentations = {'crop': Crop(tao=args.tao),
                              'mask': Mask(gamma=args.gamma),
                              'reorder': Reorder(beta=args.beta),
                              'substitute': Substitute(self.similarity_model,
                                                substitute_rate=args.substitute_rate),
                              'insert': Insert(self.similarity_model, 
                                               insert_rate=args.insert_rate,
                                               max_insert_num_per_pos=args.max_insert_num_per_pos),
                              'random': Random(tao=args.tao, gamma=args.gamma, 
                                                beta=args.beta, item_similarity_model=self.similarity_model,
                                                insert_rate=args.insert_rate, 
                                                max_insert_num_per_pos=args.max_insert_num_per_pos,
                                                substitute_rate=args.substitute_rate,
                                                augment_threshold=self.args.augment_threshold,
                                                augment_type_for_short=self.args.augment_type_for_short),
                              'combinatorial_enumerate': CombinatorialEnumerate(tao=args.tao, gamma=args.gamma, 
                                                beta=args.beta, item_similarity_model=self.similarity_model,
                                                insert_rate=args.insert_rate, 
                                                max_insert_num_per_pos=args.max_insert_num_per_pos,
                                                substitute_rate=args.substitute_rate, n_views=args.n_views)
                            }
        if self.args.base_augment_type not in self.augmentations:
            raise ValueError(f"augmentation type: '{self.args.base_augment_type}' is invalided")
        print(f"Creating Contrastive Learning Dataset using '{self.args.base_augment_type}' data augmentation")
        self.base_transform = self.augmentations[self.args.base_augment_type]
        # number of augmentations for each sequences, current support two
        self.n_views = self.args.n_views

    def _one_pair_data_augmentation(self, input_ids, ratings):
        '''
        provides two positive samples given one sequence
        '''
        if self.all is False:
            augmented_seqs = []
            for i in range(2):
                augmented_input_ids = self.base_transform(input_ids, ratings, False)
                pad_len = self.max_len - len(augmented_input_ids)
                augmented_input_ids = [0] * pad_len + augmented_input_ids
                augmented_input_ids = augmented_input_ids[-self.max_len:]
                assert len(augmented_input_ids) == self.max_len

                cur_tensors = (
                    torch.tensor(augmented_input_ids, dtype=torch.long)
                    )
                augmented_seqs.append(cur_tensors)

            return augmented_seqs

        augmented_seqs = self.base_transform(input_ids,ratings, True)

        def get_mean_vector(augmented_seqs, model, vector_size):
            vectors = []
            for seq in augmented_seqs:
                vector_seq = []
                for word in seq:
                    if str(word) in model.wv.key_to_index:
                        vector_seq.append(model.wv[str(word)])
                vectors.append(vector_seq)

            if vectors:
                vec_mean = [np.mean(vector, axis=0) for vector in vectors]
                return torch.tensor(np.array(vec_mean), device='cuda:0')
            else:
                return torch.tensor(np.zeros(len(augmented_seqs)*vector_size), device='cuda:0')

        # 找到与给定序列最相似的两个序列
        def find_most_similar_sequences(target_seq, augmented_seqs, model):
            vector_size =100 # 填写向量的维度
            target_vector = get_mean_vector([target_seq], model, vector_size)
            all_vectors = get_mean_vector(augmented_seqs, model, vector_size)

            if len(all_vectors) > 1:
                similarities = torch.cosine_similarity(target_vector, all_vectors)
                most_similar_indices = torch.argsort(similarities, dim=0)[-2:]  # 找到相似度最高的两个序列的索引
                # most_similar_indices = torch.argsort(similarities, dim=0)[:2]
                most_similar_seqs = [augmented_seqs[i] for i in most_similar_indices]
                return most_similar_seqs
            else:
                return []

        most_similar_seqs = find_most_similar_sequences(input_ids, augmented_seqs, self.model)

        augmented_seqs = []
        for augmented_seq in most_similar_seqs:
            pad_len = self.max_len - len(augmented_seq)
            padded_seq = [0] * pad_len + augmented_seq
            padded_seq = padded_seq[-self.max_len:]
            assert len(padded_seq) == self.max_len
            cur_tensors = (
                torch.tensor(padded_seq, dtype=torch.long, device='cuda:0')
            )
            augmented_seqs.append(cur_tensors)

        return augmented_seqs


    def _data_sample_rec_task(self, user_id, items, items1, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)

        for _ in copied_input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(copied_input_ids)
        copied_input_ids = [0] * pad_len + copied_input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        copied_input_ids = copied_input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(copied_input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(copied_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(copied_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_rec_tensors

    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio*len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k = insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size-2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size-2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]
        items1 = self.ratings[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0] # no use
            ratings = items1[:-3]

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]
            ratings = items1[:-2]

        else:
            items_with_noise = self._add_noise_interactions(items)
            items_with_noise1 = self._add_noise_interactions(items1)
            input_ids = items_with_noise[:-1]
            target_pos = items_with_noise[1:]
            answer = [items_with_noise[-1]]
            ratings = [items_with_noise1[:-1]]

        if self.data_type == "train":
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, items1, input_ids, \
                                            target_pos, answer)
            cf_tensors_list = []
            # if n_views == 2, then it's downgraded to pair-wise contrastive learning
            total_augmentaion_pairs = nCr(self.n_views, 2)
            for i in range(total_augmentaion_pairs):
                # print(f'inputs:{input_ids}\tratings:{ratings}\none_pair_data_augmentation:{self._one_pair_data_augmentation(input_ids,ratings)}')
                # print(len(cf_tensors_list))
                cf_tensors_list.append(self._one_pair_data_augmentation(input_ids, ratings))
                # print('cf_tensors_list:',cf_tensors_list)
            return (cur_rec_tensors, cf_tensors_list)
        elif self.data_type == 'valid':
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, items1, input_ids, target_pos, answer)
            return cur_rec_tensors
        else:
            cur_rec_tensors = self._data_sample_rec_task(user_id, items_with_noise, items_with_noise1, input_ids, \
                                target_pos, answer)
            return cur_rec_tensors

    def __len__(self):
        '''
        consider n_view of a single sequence as one sample
        '''
        return len(self.user_seq)


class SASRecDataset(Dataset):

    def __init__(self, args, user_seq, ratings, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.ratings = ratings
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in copied_input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(copied_input_ids)
        copied_input_ids = [0] * pad_len + copied_input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg


        copied_input_ids = copied_input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]


        assert len(copied_input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len


        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(copied_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(copied_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),

            )

        return cur_rec_tensors

    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        return self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)

    def __len__(self):
        return len(self.user_seq)
