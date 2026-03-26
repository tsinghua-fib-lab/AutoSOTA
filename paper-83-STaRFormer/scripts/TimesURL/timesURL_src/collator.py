from argparse import Namespace
import numpy as np, math
import random
import torch
from dataclasses import dataclass


@dataclass
class CLDataCollator:
    max_len: int
    args: Namespace
    len_sampling_bound = [0.3, 0.7]
    dense_sampling_bound = [0.4, 0.6]
    pretrain_tasks = 'full2'

    # mask_ratio_per_seg = 0.15
    # segment_num = 1
    # pretrain_tasks = 'full2'

    def __call__(self, batch):

        batch_size = len(batch)
        D = batch[0][0].size(1)

        time_batch = torch.zeros([batch_size, 2, self.max_len])
        value_batch = torch.zeros([batch_size, 2, self.max_len, D])
        if self.pretrain_tasks == 'full2':
            mask_batch = torch.zeros([batch_size, 2, self.max_len, 2 * D])
        else:
            mask_batch = torch.zeros([batch_size, 2, self.max_len, D])

        mask_old_batch = torch.zeros([batch_size, 2, self.max_len, D])
        for idx, instance in enumerate(batch):
            seq1, seq2 = self._per_seq_sampling(instance)

            v1, t1, m1, m1_old = seq1
            v2, t2, m2, m2_old = seq2

            len1 = v1.size(0)
            len2 = v2.size(0)

            # print(len1, len2)
            # print(v1.shape, t1.shape, m1.shape, v2.shape, t2.shape, m2.shape)

            value_batch[idx, 0, :len1] = v1
            time_batch[idx, 0, :len1] = t1
            mask_batch[idx, 0, :len1] = m1
            mask_old_batch[idx, 0, :len1] = m1_old

            value_batch[idx, 1, :len2] = v2
            time_batch[idx, 1, :len2] = t2
            mask_batch[idx, 1, :len2] = m2
            mask_old_batch[idx, 1, :len2] = m2_old

        return {'value': value_batch, 'time': time_batch, 'mask': mask_batch, 'mask_origin': mask_old_batch}

    def _per_seq_sampling(self, instance):
        '''
        - times is a 1-dimensional tensor containing T time values of observations.
        - values is a (T, D) tensor containing observed values for D variables.
        - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
        '''

        values, times, mask = instance

        # selected_indices = self._random_sampling_cl(values) # Random Anchor and Positive
        selected_indices = self._time_sensitive_cl(times)  # Anchor and Positive based on sampling density

        v1, t1, m1, v2, t2, m2 = [], [], [], [], [], []

        for idx, (v, t, m) in enumerate(zip(values, times, mask)):

            if idx in selected_indices:
                v1.append(v)
                t1.append(t)
                m1.append(m)

            else:
                v2.append(v)
                t2.append(t)
                m2.append(m)

        v1 = torch.stack(v1, dim=0)
        t1 = torch.stack(t1, dim=0)
        m1 = torch.stack(m1, dim=0)

        v2 = torch.stack(v2, dim=0)
        t2 = torch.stack(t2, dim=0)
        m2 = torch.stack(m2, dim=0)

        m1_old, m2_old = m1.clone(), m2.clone()
        if self.pretrain_tasks == 'full2':
            # print(torch.sum(m1, axis = 0))
            T, D = m1.shape

            m1 = self._seg_masking(mask=m1, timestamps=t1)
            # a = m1[ : , : D]
            # b = m1[ : , D : ]
            # c = a + b
            # print(torch.sum(c, axis = 0))

            # print(torch.sum(m2, axis = 0))
            m2 = self._seg_masking(mask=m2, timestamps=t2)
            # a = m2[ : , : D]
            # b = m2[ : , D : ]
            # c = a + b
            # print(torch.sum(c, axis = 0))

        return (v1, t1, m1, m1_old), (v2, t2, m2, m2_old)

    def _random_sampling_cl(self, values):
        indices = list(range(len(values)))
        random.shuffle(indices)

        length = int(np.random.uniform(self.len_sampling_bound[0], self.len_sampling_bound[1], 1)[0] * len(indices))
        length = max(length, 1)

        selected_indices = set(indices[: length])

        # print(indices)
        # print(length)
        # print(selected_indices)

        return selected_indices

    def _time_sensitive_cl(self, timestamps):

        times = torch.clone(timestamps)
        times = times.reshape(times.shape[0])

        # compute average of pre- and post- interval time for each timestep, except the first and last
        avg_interval_times = [(((times[i] - times[i - 1]) + (times[i + 1] - times[i])) / 2) for i in
                              range(1, times.shape[0] - 1)]
        avg_interval_times.append(times[-1] - times[-2])  # pre-interval time for last timestep becomes its average
        avg_interval_times.insert(0, times[1] - times[0])  # post-interval time for first timestep becomes its average
        # print(avg_interval_times)

        # sort the interval times and save its corresponding index, timestep
        # after sorting, the first section would contain the lowest interval times -> dense regions of the sample
        # last section would contain the highest interval times -> sparse regions of the sample
        pairs = [(idx, time, avg_interval_time) for idx, (time, avg_interval_time) in
                 enumerate(zip(times, avg_interval_times))]
        # print(pairs)
        pairs.sort(key=lambda pairs: pairs[2])
        indices = [idx for idx, time, avg_interval_time in pairs]
        # print(pairs)

        # length of the anchor/positive sample
        length = int(np.random.uniform(self.len_sampling_bound[0], self.len_sampling_bound[1], 1)[0] * times.shape[0])
        length = max(length, 1)
        # print(length)

        # select the indices with the most dense sampling frequency, i.e. minimum time interval
        # selected_indices = set([idx for idx, time, avg_interval_time in pairs[ : length]])
        # print(selected_indices)

        # alternate between dense and sparse sample, i.e. samples located in dense and sparse regions
        '''
        front, end = 0, len(pairs) - 1
        selected_indices = []
        for i in range(length):
            if i % 2 == 0:
                selected_indices.append(pairs[front][0])
                front += 2
            else:
                selected_indices.append(pairs[end][0])
                end -= 2
        '''

        # divide samples in pairs into two regions -> sparse (50%) and dense(50%)
        # sample a fraction, f, of the samples from the dense and the remaining, (1-f), of the samples from the sparse region
        dense_indices = indices[: int(len(indices) / 2)]
        random.shuffle(dense_indices)
        sparse_indices = indices[int(len(indices) / 2):]
        random.shuffle(sparse_indices)

        # 5 - random dense, random sparse CL
        dense_length = int(np.random.uniform(self.dense_sampling_bound[0], self.dense_sampling_bound[1], 1)[0] * length)
        dense_length = max(dense_length, 1)
        sparse_length = length - dense_length

        # 6 - 50% dense, 50% sparse CL
        # dense_length = int(0.5 * length)
        # sparse_length = length - dense_length

        selected_dense_indices = dense_indices[: dense_length]
        selected_sparse_indices = sparse_indices[: sparse_length]
        selected_dense_indices.extend(selected_sparse_indices)
        selected_indices = set(selected_dense_indices)

        return selected_indices

    def _seg_masking(self, mask=None, timestamps=None):

        '''
        - mask is a (T, D) tensor
        - timestamps is a (T, 1) tensor
        - return: (T, 2*D) tensor
        '''

        D = mask.size(1)
        interp_mask = torch.zeros_like(mask)

        for dim in range(D):
            # print('Dimension: ' + str(dim))

            # length = mask[:, dim].sum().long().item()
            # print(length)

            # length of each masked segment is constant
            # seg_pos = self._constant_length_sampling(mask[ : , dim])

            # time of each masked segment is constant: length of each masked segment may vary depending on the density of the sample in the masked region
            seg_pos = self._time_sensitive_sampling(mask[:, dim], timestamps)

            # print(mask[ : , dim])
            # print(interp_mask[ : , dim])
            # print(seg_pos)
            if len(seg_pos) > 0:
                mask[seg_pos, dim] = 0.0
                interp_mask[seg_pos, dim] = 1.0
            # print(mask[ : , dim])
            # print(interp_mask[ : , dim])

        return torch.cat([mask, interp_mask], dim=-1)

    def _constant_length_sampling(self, mask):

        # mask = torch.tensor([0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0])
        count_ones = mask.sum().long().item()

        if self.args.mask_ratio_per_seg * count_ones < 1:
            seg_seq_len = 1
        else:
            seg_seq_len = int(self.args.mask_ratio_per_seg * count_ones)

        ones_indices_in_mask = torch.where(mask == 1)[0].tolist()

        # if seg_seq_len == 1: indices = list(range(len(ones_indices_in_mask)))
        # else: indices = list(range(len(ones_indices_in_mask[ : -seg_seq_len + 1])))

        # print('mask: ' + str(mask))
        # print('count_ones: ' + str(count_ones))
        # print('seg_seq_len: ' + str(seg_seq_len))
        # print('ones_indices_in_mask: ' + str(ones_indices_in_mask))
        # print('indices: ' + str(indices))

        seg_pos = []
        for seg in range(self.args.segment_num):
            # print()
            # print(ones_indices_in_mask)

            if len(ones_indices_in_mask) > 1:
                if seg_seq_len == 1:
                    start_idx_in_mask = random.choice(ones_indices_in_mask)
                else:
                    start_idx_in_mask = random.choice(ones_indices_in_mask[: -seg_seq_len + 1])
                # print(start_idx_in_mask)

                start = ones_indices_in_mask.index(start_idx_in_mask)
                end = start + seg_seq_len

                sub_seg = ones_indices_in_mask[start: end]
                # print(sub_seg)

                seg_pos.extend(sub_seg)
                ones_indices_in_mask = list(set(ones_indices_in_mask) - set(sub_seg))
                ones_indices_in_mask.sort()

        # print('seg_pos: ' + str(seg_pos))
        return list(set(seg_pos))

    def _time_sensitive_sampling(self, mask, timestamps):

        # segment_num = 3
        # mask_ratio_per_seg = 0.15

        timestamps = timestamps.reshape(timestamps.shape[0])
        # sampled_times = timestamps[mask].tolist() # times at which this feature was sampled
        sampled_times = [timestamps[i].item() for i in range(mask.shape[0]) if mask[i] == 1]

        if len(sampled_times) == 0: return []

        # print('timestamps: ' + str(timestamps))
        # print('mask: ' + str(mask))
        # print('sampled_times: ' + str(sampled_times))
        sampled_times_start, sampled_times_end = sampled_times[0], sampled_times[-1]

        # full time interval of the feature = last sampling time - first sampling time
        # time of masked segment = a fixed percentage of the full time interval of the feature
        time_of_masked_segment = (sampled_times_end - sampled_times_start) * self.args.mask_ratio_per_seg
        # print('time_of_masked_segment: ' + str(time_of_masked_segment))

        available_samples_to_sample = [time for time in sampled_times if
                                       time < sampled_times_end - time_of_masked_segment]
        # print('available_samples_to_sample: ' + str(available_samples_to_sample))

        if len(available_samples_to_sample) > 0:
            chosen_time = random.choice(available_samples_to_sample)
        else:
            return []
        # print('chosen_time: ' + str(chosen_time))

        masking_times = []
        for i in range(self.args.segment_num):

            masked_segment_start_time = chosen_time
            masked_segment_end_time = masked_segment_start_time + time_of_masked_segment

            idx = sampled_times.index(chosen_time)
            chosen_times = [chosen_time]
            available_samples_to_sample.remove(chosen_time)

            for time in sampled_times[idx + 1:]:
                if time > masked_segment_end_time:
                    break

                if masked_segment_start_time < time and time <= masked_segment_end_time:
                    chosen_times.append(time)

                if time in available_samples_to_sample:
                    available_samples_to_sample.remove(time)
                # print('           available_samples_to_sample: ' + str(available_samples_to_sample))

            masking_times.extend(chosen_times)

            for time in sampled_times[: idx][::-1]:
                if time < chosen_time - time_of_masked_segment or time > chosen_time + time_of_masked_segment:
                    break

                if time > chosen_time - time_of_masked_segment and time < chosen_time + time_of_masked_segment and time in available_samples_to_sample:
                    available_samples_to_sample.remove(time)

            if len(available_samples_to_sample) > 0:
                chosen_time = random.choice(available_samples_to_sample)
            else:
                return []
            # print('chosen_times: ' + str(chosen_times))
            # print('available_samples_to_sample: ' + str(available_samples_to_sample))
            # print('chosen_time: ' + str(chosen_time))

        times = timestamps.tolist()
        seg_pos = [times.index(time) for time in masking_times]
        # print('masking_times: ' + str(masking_times))
        # print('seg_pos: ' + str(seg_pos))
        return list(set(seg_pos))

    '''
    def _seg_sampling(self, max_len):
        if max_len * self.args.mask_ratio_per_seg < 1:
            return []
        seg_pos = []
        seg_len = int(max_len * self.args.mask_ratio_per_seg)
        print('seg_len: ' + str(seg_len))
        start_pos = np.random.randint(max_len, size=self.args.segment_num)
        print('start_pos: ' + str(start_pos))
        for start in start_pos:
            seg_pos += list(range(start, min(start+seg_len, max_len)))
        print(seg_pos)
        return seg_pos
    '''


# ---Test _time_sensitive_sampling function for reconstruction task---#
'''
m = torch.zeros((56), dtype = bool)
l = [3, 8, 11, 13, 18, 19, 42, 45, 50, 52, 55]
m[l] = 1
t = torch.zeros((56), dtype = float)
times = torch.tensor([1, 5, 8, 9, 12, 13, 17, 20, 23, 28, 31], dtype = float)
t[l] = times
# print(m)
# print(t)
train_cl_collator = CLDataCollator(max_len = 50)
train_cl_collator._time_sensitive_sampling(m, t)
'''

# ----------Test _time_sensitive_cl function for CL task----------#
'''
times = torch.tensor([1, 2, 3, 4, 5, 15, 18, 25, 26, 27, 28, 29, 35, 45])
times = times.reshape(times.shape[0], 1)
train_cl_collator = CLDataCollator(max_len = 50)
selected_indices = train_cl_collator._time_sensitive_cl(times)
'''

'''
max_len = 50
D = 4
value, time, mask = torch.rand(max_len, D), torch.rand(max_len, 1), torch.randint(0, 2, (max_len, D))
data = [value, time, mask]
batch = [data]
train_cl_collator = CLDataCollator(max_len = max_len)
# (v1, t1, m1), (v2, t2, m2) = train_cl_collator._per_seq_sampling(data)
# print(v1.shape, t1.shape, m1.shape, v2.shape, t2.shape, m2.shape)
out = train_cl_collator.__call__(batch)
'''

'''
print(out['value'].shape, out['time'].shape, out['mask'].shape)
print('Value')
print(value)
print(out['value'][0, 0].shape)
print(out['value'][0, 1].shape)
print('Time')
print(time)
print(out['time'][0, 0].shape)
print(out['time'][0, 1].shape)
print('Mask')
print(mask)
print(out['mask'][0, 0])
print(out['mask'][0, 1])
print(torch.sum(mask, axis = 0))
print(torch.sum(out['mask'][0, 0], axis = 0))
print(torch.sum(out['mask'][0, 1], axis = 0))
'''