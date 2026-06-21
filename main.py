import argparse
import json
import librosa
import numpy as np
import os
import tqdm
import warnings
import torch
from pydub import AudioSegment
import auditok
from audiobox_aesthetics.infer import initialize_predictor
import torchaudio
import copy
from concurrent.futures import ThreadPoolExecutor
import random
from msclap import CLAP
from utils.tool import get_audio_files
from utils.logger import Logger, time_logger
from models.beats.BEATs import BEATs, BEATsConfig
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore")
audio_count = 0


@time_logger
def standardization(audio):
    """
    Preprocess the audio file, including setting sample rate, bit depth, channels, and volume normalization.

    Args:
        audio (str or AudioSegment): Audio file path or AudioSegment object, the audio to be preprocessed.

    Returns:
        dict: A dictionary containing the preprocessed audio waveform, audio file name, and sample rate, formatted as:
              {
                  "waveform": np.ndarray, the preprocessed audio waveform, dtype is np.float32, shape is (num_samples,)
                  "name": str, the audio file name
                  "sample_rate": int, the audio sample rate
              }

    Raises:
        ValueError: If the audio parameter is neither a str nor an AudioSegment.
    """
    global audio_count
    name = "audio"

    if isinstance(audio, str):
        name = os.path.basename(audio)
        audio = AudioSegment.from_file(audio)
    elif isinstance(audio, AudioSegment):
        name = f"audio_{audio_count}"
        audio_count += 1
    else:
        raise ValueError("Invalid audio type")

    logger.debug("Entering the preprocessing of audio")

    # Convert the audio file to WAV format
    audio = audio.set_frame_rate(24000)
    audio = audio.set_sample_width(2)  # Set bit depth to 16bit
    audio = audio.set_channels(1)  # Set to mono

    logger.debug("Audio file converted to WAV format")

    # Calculate the gain to be applied
    target_dBFS = -20
    gain = target_dBFS - audio.dBFS
    logger.info(f"Calculating the gain needed for the audio: {gain} dB")

    # Normalize volume and limit gain range to between -3 and 3
    normalized_audio = audio.apply_gain(min(max(gain, -3), 3))

    waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)
    max_amplitude = np.max(np.abs(waveform))
    waveform /= max_amplitude  # Normalize

    logger.debug(f"waveform shape: {waveform.shape}")
    logger.debug("waveform in np ndarray, dtype=" + str(waveform.dtype))

    return {
        "waveform": waveform,
        "name": name,
        "sample_rate": 24000,
    }


@time_logger
def audio_activity_detection(audio_path):
    """
    Perform Audio Activity Detection (AAD) on the given audio.
    Args:
        audio_path (dict): path of the audio which will be detected.
    Returns:
        events (list): A list of tuples containing the start and end times of detected audio activities.
    """
    audio_events = auditok.split(
        audio_path,
        min_dur=1.2,          # Minimum duration of a valid audio event in seconds, default 0.2s
        max_dur=30,         # Maximum duration of an event, default 4s
        max_silence=1,    # Maximum tolerated silence duration within an event
        energy_threshold=55 # Detection threshold
    )

    events = []
    for i, r in enumerate(audio_events):
        # AudioRegions returned by `split` have defined 'start' and 'end' attributes
        events.append((r.start, r.end))
    
    return events


@time_logger
def audio_event_detection(audio, events, win_len=5, win_hop=3, threshold=0.6):
    """
    Detect the audio events using the given model.

    Args:
        audio (str or dict): The audio file path or a dictionary containing audio waveform and sample rate.
        events (list): A list of tuples containing the start and end times of detected audio activities.
        win_len (int): The window length(s) for detecting.
        win_hop (int): The window hop(s) for detecting.
        threshold (float): The threshold for detecting.
    Returns:
        dict: A dictionary containing the audio waveform, audio file name, sample rate, and AED results.
    """

    # convert audio format
    waveform, rate = None, None
    win_len_frame = int(win_len * 16000)
    win_hop_frame = int(win_hop * 16000)

    if isinstance(audio, str):
        waveform, rate = librosa.load(audio, mono=False, sr=16000)
    else:
        # resample to 16000
        rate = audio["sample_rate"]
        waveform = librosa.resample(audio["waveform"], orig_sr=rate, target_sr=16000)
    
    duration = len(waveform) / 16000
    waveform = torch.tensor(waveform)
    aed_result = []
    
    # forloop each audio activity segment
    for event in events:
        event_result = []
        start_time = event[0]
        end_time = event[1]
        start_frame = int(start_time * 16000)
        end_frame = int(end_time * 16000)
        waveform_activity = waveform[start_frame:end_frame]  # get the audio activity segment
        waveform_activity = waveform_activity.to(torch.device("cuda:1"))
        segment_duration = len(waveform_activity) / 16000

        # predict the classification probability of each class
        if segment_duration <= win_len and segment_duration >= 1.2:
            with torch.inference_mode():
                probs = BEATs_model.extract_features(torch.unsqueeze(waveform_activity, 0))[0]  # torch.Size([1, 527])
            for idx, (top1_label_prob, top1_label_idx) in enumerate(zip(*probs.topk(k=1))):
                top1_label = [checkpoint['label_dict'][label_idx.item()] for label_idx in top1_label_idx]
                # print(f'Top 3 predicted labels of the {i}th audio are {top3_label} with probability of {top3_label_prob}')
                if top1_label_prob[0] > threshold:
                    top1_label = id_class_mapping[top1_label[0]]  # convert ontology id to class name
                    event_result.append([top1_label, start_time, end_time])
        elif segment_duration > win_len:
            # 窗长为 win_len，窗移为 win_hop，遍历音频进行检测
            # print(waveform.shape[1])
            # print(torch.unsqueeze(waveform[0][:160000], 0).shape)
            for i in range(0, waveform_activity.shape[0], win_hop_frame):
                if i + win_len_frame <= waveform_activity.shape[0]:
                    # 剩余时长大于等于 检测窗长
                    with torch.inference_mode():
                        probs = BEATs_model.extract_features(torch.unsqueeze(waveform_activity[i: i+win_len_frame], 0))[0]  # torch.Size([1, 527])
                    for idx, (top1_label_prob, top1_label_idx) in enumerate(zip(*probs.topk(k=1))):
                        top1_label = [checkpoint['label_dict'][label_idx.item()] for label_idx in top1_label_idx]
                        # print(f'Top 3 predicted labels of the {i}th audio are {top3_label} with probability of {top3_label_prob}')
                    if top1_label_prob[0] > threshold:
                        top1_label = id_class_mapping[top1_label[0]]
                        event_result.append([top1_label, start_time + i / 16000, start_time + (i + win_len_frame) / 16000])
                elif waveform_activity.shape[0] - i >= 19200:
                    # 剩余时长小于 检测窗长 且大于音频事件临界时长（默认1.2s）
                    with torch.inference_mode():
                        probs = BEATs_model.extract_features(torch.unsqueeze(waveform_activity[i:], 0))[0]  # torch.Size([1, 527])
                    for idx, (top1_label_prob, top1_label_idx) in enumerate(zip(*probs.topk(k=1))):
                        top1_label = [checkpoint['label_dict'][label_idx.item()] for label_idx in top1_label_idx]
                        # print(f'Top 3 predicted labels of the {i}th audio are {top3_label} with probability of {top3_label_prob}')
                    if top1_label_prob[0] > threshold:
                        top1_label = id_class_mapping[top1_label[0]]
                        event_result.append([top1_label, start_time + i / 16000, end_time])
                    break  # 避免因窗移小于窗长导致的重复检测最后一段
        if len(event_result) != 0:
            aed_result.append([event, event_result])

    # 后处理事件检测段
    # 在同一个有声段内，若相邻的事件检测段判断的TOP1事件相同，则拼接为一个检测段
    concatenated_aed_result = []
    for result in aed_result:
        # 同一个有声段的事件检测段不一定是相邻的
        event_time = result[0]
        start_time = event_time[0]
        end_time = event_time[1]
        event_labels_timestamps = result[1]

        segment_top1_label = event_labels_timestamps[0][0]  # 第一个检测段的TOP1事件标签
        segment_start_time = event_labels_timestamps[0][1]  # 第一个检测段的开始时间
        segment_end_time = event_labels_timestamps[0][2]    # 第一个检测段的结束时间
        for i in range(1, len(event_labels_timestamps)):
            if segment_end_time >= event_labels_timestamps[i][1]:
                # 下一个检测段与当前检测段相邻
                # 进行拼接判断
                if event_labels_timestamps[i][0] != segment_top1_label:
                    # 下一个检测段与当前检测段的标签不一致
                    # 保存当前检测段
                    concatenated_aed_result.append([(segment_start_time, segment_end_time), segment_top1_label])

                    # label、开始、结束指针指向下一个检测段
                    segment_top1_label = event_labels_timestamps[i][0]
                    segment_start_time = event_labels_timestamps[i][1]
                    segment_end_time = event_labels_timestamps[i][2]
                
                else:
                    # 下一个检测段与当前检测段的标签一致
                    # 结束指针指向下一个检测段
                    segment_end_time = event_labels_timestamps[i][2]
            else:
                # 下一个检测段与当前检测段不相邻
                # 不进行拼接，保存当前检测段
                concatenated_aed_result.append([(segment_start_time, segment_end_time), segment_top1_label])

                # label、开始、结束指针指向下一个检测段
                segment_top1_label = event_labels_timestamps[i][0]
                segment_start_time = event_labels_timestamps[i][1]
                segment_end_time = event_labels_timestamps[i][2]

        # 保存最后一个检测段
        concatenated_aed_result.append([(segment_start_time, segment_end_time), segment_top1_label]) 

    # 重检测事件检测段
    # 遍历拼接后的事件检测段
    re_concatenated_aed_result = []
    for segment_time_label in concatenated_aed_result:
        segment_start_time = segment_time_label[0][0]
        segment_end_time = segment_time_label[0][1]
        segment_label = [segment_time_label[1]]  # 检测段的类别标签以列表格式进行存储

        if segment_end_time - segment_start_time <= win_len:
            # 若当前事件检测段的时长小于等于检测窗长
            # 则说明是非拼接段，不需要进行重检测
            re_concatenated_aed_result.append([segment_time_label[0], segment_label])
            continue
        
        start_frame = int(segment_start_time * 16000)
        end_frame = int(segment_end_time * 16000)
        waveform_activity = waveform[start_frame:end_frame]  # get the audio activity segment
        waveform_activity = waveform_activity.to(torch.device("cuda:1"))
        segment_duration = len(waveform_activity) / 16000
        
        # 再次使用 AS-2M finetuned BEATs model 进行事件检测
        with torch.inference_mode():
            probs = BEATs_model.extract_features(torch.unsqueeze(waveform_activity, 0))[0]  # torch.Size([1, 527])
        for idx, (topk_label_prob, topk_label_idx) in enumerate(zip(*probs.topk(k=2))):
            topk_label = [checkpoint['label_dict'][label_idx.item()] for label_idx in topk_label_idx]
            top1_label = id_class_mapping[topk_label[0]]  # convert ontology id to class name
            top2_label = id_class_mapping[topk_label[1]]
            # print(f"topk_label: {topk_label}, topk_label_prob: {topk_label_prob}, old_topk_label: {segment_time_label[1]}")
            if topk_label_prob[1] > threshold and top2_label != segment_time_label[1]:
                # 若判断的TOP2类别概率超过阈值，且与原本的伪标签不同
                # 则该样本的标签增添重检测得到的TOP2类别
                segment_label.append(top2_label)
                
                if top1_label != segment_time_label[1]:
                    # 且判断TOP1类别是否与原本的伪标签不同
                    # 若不相同则增添TOP1类别
                    segment_label.append(top1_label)
            
            elif topk_label_prob[0] > threshold and top1_label != segment_time_label[1]:
                #  TOP2类别不满足要求，判断TOP1类别
                segment_label.append(top1_label)
        
        re_concatenated_aed_result.append([segment_time_label[0], segment_label])

    audio["concatenated_aed_result"] = re_concatenated_aed_result
    return audio


@time_logger
def audiobox_aesthetics_prediction(audio, aes_predictor):
    """
    params:
        audio (dict): A dictionary containing the audio waveform, audio file name, sample rate, and AED results.
        aes_predictor (model): audiobox aesthetics predictor.

    returns:
        avg_aes (list): A list containing the average CE, CU, PC, PQ score.
        aes_result (list): A list containing the result from aesthetics predictor, each element presents one clip's aesthetic.
    """
    waveform = audio["waveform"]
    ori_sr = audio["sample_rate"]
    concatenated_aed_result = audio["concatenated_aed_result"]

    # convert audio format
    tensor_waveform = torch.tensor(waveform)

    # forloop audio event detection clips and calculate the aesthetic for each clip
    aes_result = []
    CE = 0
    CU = 0
    PC = 0
    PQ = 0
    for aed_clip in concatenated_aed_result:
        start_time = aed_clip[0][0]
        end_time = aed_clip[0][1]
        start_frame = int(start_time * ori_sr)
        end_frame = int(end_time * ori_sr)

        # forward函数首先将音频重采样为16kHz，且转换为单声道，然后进行推理预测audiobox aesthetics
        tensor_waveform_segment = tensor_waveform[start_frame: end_frame]
        tensor_waveform_segment = torch.unsqueeze(tensor_waveform_segment, 0)
        aes_prediction = aes_predictor.forward([{"path": tensor_waveform_segment, "sample_rate": ori_sr}])
        CE += aes_prediction[0]["CE"]
        CU += aes_prediction[0]["CU"]
        PC += aes_prediction[0]["PC"]
        PQ += aes_prediction[0]["PQ"]
        aes_result.append(aes_prediction[0])
    
    clips_num = len(aes_result)
    avg_CE = CE / clips_num
    avg_CU = CU / clips_num
    avg_PC = PC / clips_num
    avg_PQ = PQ / clips_num
    avg_aes = [avg_CE, avg_CU, avg_PC, avg_PQ]

    return avg_aes, aes_result


@time_logger
def evaluation_metric_prediction(audio, aes_predictor, clap_predictor):
    """
    params:
        audio (dict): A dictionary containing the audio waveform, audio file name, sample rate, and AED results.
        aes_predictor (model): audiobox aesthetics predictor.
        clap_predictor (model): clap similarity predictor.

    returns:
        avg_evaluation_metric (list): A list containing the average CE, CU, PC, PQ score and the average CLAP similarity.
        evaluation_metric_result (list): A list containing the result from aesthetics predictor and clap predictor, each element presents one clip's aesthetic and clap.
    """
    waveform = audio["waveform"]
    ori_sr = audio["sample_rate"]
    # print(ori_sr)
    concatenated_aed_result = audio["concatenated_aed_result"]

    # convert audio format
    tensor_waveform = torch.tensor(waveform)

    # forloop audio event detection clips and calculate the aesthetic for each clip
    evaluation_metric_result = []
    CE = 0
    CU = 0
    PC = 0
    PQ = 0
    clap_similarity_sum = 0
    clap_num = 0
    for aed_clip in concatenated_aed_result:
        start_time = aed_clip[0][0]
        end_time = aed_clip[0][1]
        event_label = aed_clip[1]
        start_frame = int(start_time * ori_sr)
        end_frame = int(end_time * ori_sr)

        # forward函数首先将音频重采样为16kHz，且转换为单声道，然后进行推理预测audiobox aesthetics
        tensor_waveform_segment = tensor_waveform[start_frame: end_frame]
        tensor_waveform_segment = torch.unsqueeze(tensor_waveform_segment, 0)
        aes_prediction = aes_predictor.forward([{"path": tensor_waveform_segment, "sample_rate": ori_sr}])
        CE += aes_prediction[0]["CE"]
        CU += aes_prediction[0]["CU"]
        PC += aes_prediction[0]["PC"]
        PQ += aes_prediction[0]["PQ"]

        # 计算CLAP相似度
        # 将音频重采样为44.1kHz，并转换为一维Tensor
        tensor_waveform_segment_clap = torch.unsqueeze(tensor_waveform[start_frame: end_frame], 0)
        tensor_waveform_segment_clap = torchaudio.functional.resample(
            tensor_waveform_segment_clap,
            orig_freq=ori_sr,
            new_freq=44100,
        )
        tensor_waveform_segment_clap = tensor_waveform_segment_clap.reshape(-1)

        # 若音频时长不足7s，则repeat到7s，否则在这条音频中，随机截取7s的片段
        if 7 * 44100 >= tensor_waveform_segment_clap.shape[0]:
            repeat_factor = int(np.ceil((7 * 44100) / tensor_waveform_segment_clap.shape[0]))
            tensor_waveform_segment_clap = tensor_waveform_segment_clap.repeat(repeat_factor)
            tensor_waveform_segment_clap = tensor_waveform_segment_clap[:7 * 44100]
        else:
            start_index = random.randrange(tensor_waveform_segment_clap.shape[0] - 7 * 44100)
            tensor_waveform_segment_clap = tensor_waveform_segment_clap[start_index: start_index + 
                                                                        7 * 44100]
        
        # 转换为单精度浮点张量，并添加channel和batch维
        tensor_waveform_segment_clap = torch.FloatTensor(tensor_waveform_segment_clap) 
        tensor_waveform_segment_clap = tensor_waveform_segment_clap.reshape(1, -1)  # [channel, time]
        tensor_waveform_segment_clap = torch.unsqueeze(tensor_waveform_segment_clap, 0)  # [batch, channel, time]
        tensor_waveform_segment_clap = tensor_waveform_segment_clap.to(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

        # 推理提取文本嵌入和音频嵌入，并计算CLAP相似度
        with torch.inference_mode():
            text_embeddings = clap_predictor.get_text_embeddings(event_label)
            audio_embeddings = clap_predictor._get_audio_embeddings(tensor_waveform_segment_clap)
            clap_similarity = clap_predictor.compute_similarity(audio_embeddings, text_embeddings)
        clap_similarity = clap_similarity.detach().cpu().numpy()[0]  # <class 'numpy.ndarray'>: [clap_sim_1 clap_sim_2..]
        clap_num += len(clap_similarity)
        clap_similarity_sum += float(sum(sim for sim in clap_similarity))

        # 将aesthetics和CLAP的结果存储进一个列表中
        evaluation_metric_result.append([aes_prediction[0], clap_similarity])
    
    clips_num = len(evaluation_metric_result)
    avg_CE = CE / clips_num
    avg_CU = CU / clips_num
    avg_PC = PC / clips_num
    avg_PQ = PQ / clips_num
    avg_clap_similarity = clap_similarity_sum / clap_num
    avg_evaluation_metric = [avg_CE, avg_CU, avg_PC, avg_PQ, avg_clap_similarity]

    return avg_evaluation_metric, evaluation_metric_result


@time_logger
def filter(audio, aes_result, aes_threshold=[1, 1, 1.8, 5.5]):
    """
    Filter out the segments with aesthetic scores.

    Args:
        audio (dict): A dictionary containing the audio waveform, audio file name, sample rate, and AED results.
        aes_result (list): A list containing the result from aesthetics predictor, each element presents one clip's aesthetic.
        aes_threshold (list): A list containing the aesthetics threshold.

    Returns:
        audio (dict): A dictionary containing the audio waveform, audio file name, sample rate, AED results, and filtered results.
    """
    concatenated_aed_result = audio["concatenated_aed_result"]
    # CE_threshold = aes_threshold[0]
    # CU_threshold = aes_threshold[1]
    PC_threshold = aes_threshold[2]
    PQ_threshold = aes_threshold[3]
    filtered_result = []

    # 遍历检测段，过滤 aesthetics 小于所设阈值的检测段
    for i in range(len(aes_result)):
        # CE = aes_result[i]["CE"]
        # CU = aes_result[i]["CU"]
        PC = aes_result[i]["PC"]
        PQ = aes_result[i]["PQ"]
        if PC < PC_threshold or PQ < PQ_threshold:
            # 任一评估指标小于所设阈值，过滤该检测段
            continue
        
        concatenated_aed_result_deepcopy = copy.deepcopy(concatenated_aed_result[i])
        concatenated_aed_result_deepcopy.append(aes_result[i])
        filtered_result.append(concatenated_aed_result_deepcopy)
    
    audio["filtered_result"] = filtered_result
    
    return audio


@time_logger
def filter_version_b(audio, evaluation_metric_result, threshold=[1, 1, 2.24, 5.85, (7.43, 7.43)]):
    """
    Filter out the segments with aesthetic scores and CLAP similarity.

    Args:
        audio (dict): A dictionary containing the audio waveform, audio file name, sample rate, and AED results.
        evaluation_metric_result (list): A list containing the result from aesthetics predictor and clap predictor, each element presents one clip's aesthetic and clap.
        threshold (list): A list containing the aesthetics and clap threshold.

    Returns:
        audio (dict): A dictionary containing the audio waveform, audio file name, sample rate, AED results, and filtered results.
    """
    concatenated_aed_result = audio["concatenated_aed_result"]
    # CE_threshold = threshold[0]
    # CU_threshold = threshold[1]
    PC_threshold = threshold[2]
    PQ_threshold = threshold[3]
    CLAP_threshold = threshold[4]
    filtered_result = []

    # 遍历检测段，过滤 aesthetics 或 CLAP 相似度小于所设阈值的检测段
    for i in range(len(evaluation_metric_result)):
        topk_label = concatenated_aed_result[i][1]  # [top1_label, top2_label...]
        # CE = evaluation_metric_result[i][0]["CE"]
        # CU = evaluation_metric_result[i][0]["CU"]
        PC = evaluation_metric_result[i][0]["PC"]
        PQ = evaluation_metric_result[i][0]["PQ"]
        CLAP_similarity = evaluation_metric_result[i][1]  # [clap_sim_1 clap_sim_2...]
        continue_flag = False
        
        if PC < PC_threshold or PQ < PQ_threshold:
            # aesthetics 小于所设阈值，过滤该检测段
            continue
        
        for idx, top_label in enumerate(topk_label):
            # CLAP 小于所设阈值，过滤该检测段
            if top_label == "Speech" and float(CLAP_similarity[idx]) < CLAP_threshold[0]:
                continue_flag = True
            elif top_label != "Speech" and float(CLAP_similarity[idx]) < CLAP_threshold[1]:
                continue_flag = True
        
        if continue_flag:
            continue
        
        concatenated_aed_result_deepcopy = copy.deepcopy(concatenated_aed_result[i])
        concatenated_aed_result_deepcopy.append(evaluation_metric_result[i])
        filtered_result.append(concatenated_aed_result_deepcopy)
    
    audio["filtered_result"] = filtered_result
    return audio


@time_logger
def save_clips(audio, evaluation_metric_result, save_path, audio_idx):
    """
    Save audio clips from the AudioPipeline and Write the jsonl file which contains the matadata for convinient index.

    Args:
        audio (dict): A dictionary containing the audio waveform, audio file name, sample rate, AED results, and filtered results.
        evaluation_metric_result (list): 
            version a: A list containing the result from aesthetics predictor, each element presents one clip's aesthetic.
            version b: A list containing the result from aesthetics predictor and clap predictor, each element presents one clip's aesthetic and clap.
        save_path (str): The path for saving audio clips.
    """

    # save audio clips
    filtered_result = audio["filtered_result"]
    concatenated_aed_result = audio["concatenated_aed_result"]
    sr = audio["sample_rate"]
    waveform = audio["waveform"]
    name = audio["name"]
    os.makedirs(save_path, exist_ok=True)

    # Function to process each segment in a separate thread
    def process_segment(idx, segment):
        start, end = int(segment[0][0] * sr), int(segment[0][1] * sr)
        audio_clip = waveform[start:end]
        audio_clip = librosa.to_mono(audio_clip)
        file_name = f"{name}_{idx}.mp3"
        out_path = os.path.join(save_path, file_name)

        try:
            # Ensure x is in the correct format and normalize if necessary
            if audio_clip.dtype != np.int16:
                # Normalize the array to fit in int16 range if it's not already int16
                audio_clip = np.int16(audio_clip / np.max(np.abs(audio_clip)) * 32767)

            # Create audio segment from numpy array
            audio = AudioSegment(
                audio_clip.tobytes(), frame_rate=sr, sample_width=audio_clip.dtype.itemsize, channels=1
            )
            # Export as MP3 file
            audio.export(out_path, format="mp3")
        except Exception as e:
            print(e)
            print("Error: Failed to write MP3 file.")

    # Use ThreadPoolExecutor for concurrent execution
    with ThreadPoolExecutor(max_workers=18) as executor:
        # Submit each segment processing as a separate thread
        futures = [
            executor.submit(process_segment, idx, segment)
            for idx, segment in enumerate(filtered_result)
        ]

        # Wait for all threads to complete
        for future in tqdm.tqdm(
            futures, total=len(filtered_result), desc=f"Audio_idx: {audio_idx}. Exporting to MP3"
        ):
            future.result()
    
    # write jsonl file
    jsonl_path = os.path.join(save_path, name + ".json")
    if isinstance(filtered_result[0][-1], list):
        # version b
        # filtered result
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for idx, item in enumerate(filtered_result):
                f.write(json.dumps({"id": name + f"_{idx}", "path": name + f"_{idx}.mp3", 
                                    "label": item[1], "aesthetics": item[2][0], "clap_similarity": item[2][1].tolist(), 
                                    "duration": item[0][1] - item[0][0]},  
                                    ensure_ascii=False) + "\n")
        
        # result before filtering
        jsonl_path = os.path.join(save_path, name + "_before_filter.json")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for idx, item in enumerate(evaluation_metric_result):
                f.write(json.dumps({"id": name + f"_{idx}", 
                                    "label": concatenated_aed_result[idx][1], "aesthetics": item[0], "clap_similarity": item[1].tolist(), 
                                    "duration": concatenated_aed_result[idx][0][1] - concatenated_aed_result[idx][0][0]},  
                                    ensure_ascii=False) + "\n")
    else:
        # version a
        # filtered result
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for idx, item in enumerate(filtered_result):
                f.write(json.dumps({"id": name + f"_{idx}", "path": name + f"_{idx}.mp3", "label": item[1], "aesthetics": item[2], 
                                    "duration": item[0][1] - item[0][0]}, 
                                    ensure_ascii=False) + "\n")
        # result before filtering
        jsonl_path = os.path.join(save_path, name + "_before_filter.json")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for idx, item in enumerate(evaluation_metric_result):
                f.write(json.dumps({"id": name + f"_{idx}", 
                                    "label": concatenated_aed_result[idx][1], "aesthetics": item, 
                                    "duration": concatenated_aed_result[idx][0][1] - concatenated_aed_result[idx][0][0]},  
                                    ensure_ascii=False) + "\n")


def main_process(audio_path, audio_idx=None, save_path=None, audio_name=None, args=None):
    """
    Process the audio file, including standardization, audio activity detection, audio event detection, filtering, export to MP3 and jsonl.

    Args:
        audio_path (str): Audio file path.
        save_path (str, optional): Save path, defaults to None, which means saving in the "_processed" folder in the audio file's directory.
        audio_name (str, optional): Audio file name, defaults to None, which means using the file name from the audio file path.
        audio_idx (int): Audio index.
        args (argparse.Namespace): The arguments from command line.

    Return:
        None
    """
    if not audio_path.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac")):
        logger.warning(f"Unsupported file type: {audio_path}")

    # for a single audio from path aaa/bbb/ccc.wav ---> save to aaa/bbb_processed/ccc/ccc_0.wav
    audio_name = audio_name or os.path.splitext(os.path.basename(audio_path))[0]
    save_path = save_path or os.path.join(
        os.path.dirname(audio_path) + "_processed", audio_name
    )
    logger.debug(
        f"Processing audio: {audio_name}, from {audio_path}, save to: {save_path}, audio_idx: {audio_idx}"
    )

    logger.info(
        f"Audio_idx: {audio_idx}. Step 1: Preprocess all audio files --> 24k sample rate + wave format + loudnorm + bit depth 16"
    )
    audio = standardization(audio_path)

    logger.info(f"Audio_idx: {audio_idx}. Step 2: Audio Activity Detection (AAD)")
    events = audio_activity_detection(audio_path)

    if len(events) == 0:
        # if AAD can not find any audio activities
        name = audio["name"]
        logger.info(f"Audio_idx: {audio_idx}. [{name}] AAD module found zero audio activity")
        return None
    
    logger.info(f"Audio_idx: {audio_idx}. Step 3: Audio Event Detection (AED)")
    audio = audio_event_detection(audio, events)
    logger.info(f"Audio_idx: {audio_idx}. Step 3: Audio Event Detection (AED) Finish!")
    if len(audio["concatenated_aed_result"]) == 0:
        # if the length of AED result is zero
        name = audio["name"]
        logger.info(f"Audio_idx: {audio_idx}. [{name}] the length of AED result is zero")
        return None

    logger.info(f"Audio_idx: {audio_idx}. Step 4: Filter")
    # logger.info("Step 4.1a: calculate audiobox aesthetics")
    # avg_aes, aes_result = audiobox_aesthetics_prediction(audio, aes_predictor)

    # logger.info(f"Step 4.1a: done, average CE: {avg_aes[0]}, average CU: {avg_aes[1]}, average PC: {avg_aes[2]} average PQ: {avg_aes[3]}")
    # # print(len(aes_result), len(audio["concatenated_aed_result"]))

    # logger.info("Step 4.2a: Filter out segments with less than fixed aes threshold")
    # audio = filter(audio, aes_result)
    # evaluation_metric_result = aes_result

    logger.info(f"Audio_idx: {audio_idx}. Step 4.1b: calculate audiobox aesthetics and CLAP similarity")
    avg_evaluation_metric, evaluation_metric_result = evaluation_metric_prediction(audio, aes_predictor, clap_predictor)

    logger.info(f"Audio_idx: {audio_idx}. Step 4.1b: done, average CE: {avg_evaluation_metric[0]}, average CU: {avg_evaluation_metric[1]}, \
                average PC: {avg_evaluation_metric[2]}, average PQ: {avg_evaluation_metric[3]}, average CLAP: {avg_evaluation_metric[4]}")

    logger.info(f"Audio_idx: {audio_idx}. Step 4.2b: Filter out segments with less than fixed aes and clap threshold")
    filtering_threshold = [1, 1, args.pc_threshold, args.pq_threshold, (args.clap_threshold, args.clap_threshold)]
    audio = filter_version_b(audio, evaluation_metric_result, threshold=filtering_threshold)

    if len(audio["filtered_result"]) == 0:
        # if the length of filtered result is zero
        name = audio["name"]
        logger.info(f"Audio_idx: {audio_idx}. [{name}]: the length of filtered result is zero")
        return None

    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Audio_idx: {audio_idx}. Step 5: write result into MP3 and JSON file")
    save_clips(audio, evaluation_metric_result, save_path, audio_idx)

    final_path = os.path.join(save_path, audio_name + ".json")
    logger.info(f"Audio_idx: {audio_idx}. All done, Saved to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path",
        type=str,
        default="",
        help="input folder path",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=10,
        help="The number of parallel processing threads.",
    )
    parser.add_argument(
        "--pc_threshold",
        type=float,
        default=2.24,
        help="PC threshold for filtering",
    )
    parser.add_argument(
        "--pq_threshold",
        type=float,
        default=5.85,
        help="PQ threshold for filtering",
    )
    parser.add_argument(
        "--clap_threshold",
        type=float,
        default=7.43,
        help="CLAP threshold for filtering",
    )
    args = parser.parse_args()

    logger = Logger.get_logger()

    if torch.cuda.is_available():
        torch_device = torch.device("cuda:0")
        logger.debug(f"Using GPU: 0")
    else:
        torch_device = torch.device("cpu")
        logger.debug("Using CPU")

    ## Load models
    logger.debug("Loading models...")

    # Audio Event Detection
    logger.debug(" * Loading AED Model")
    checkpoint = torch.load('./ckpt/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt', map_location=torch_device)
    beats_cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(beats_cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.to(torch_device)
    BEATs_model.eval()
    with open("ontology_id_class_map.json", "r") as f:
        id_class_mapping = json.load(f)
        f.close()

    # Audiobox Aesthetics Prediction
    logger.debug(" * Loading Audiobox Aesthetics Model")
    aes_predictor = initialize_predictor(ckpt="./ckpt/audiobox-aesthetics/checkpoint.pt")

    # CLAP Similarity Prediction
    logger.debug(" * Loading CLAP Model")
    clap_predictor = CLAP("./ckpt/CLAP_weights_2023.pth", version = '2023', use_cuda=True)

    logger.debug("All models loaded")

    input_folder_path = args.input_folder_path
    if not os.path.exists(input_folder_path):
        raise FileNotFoundError(f"input_folder_path: {input_folder_path} not found")

    audio_paths = get_audio_files(input_folder_path)  # Get all audio files
    logger.debug(f"Scanning {len(audio_paths)} audio files in {input_folder_path}")

    # multithreaded parallel processing
    with ThreadPoolExecutor(max_workers=args.threads, thread_name_prefix="outer") as outer_executor:
        outer_futures = [
            outer_executor.submit(main_process, path, audio_idx, args=args)
            for audio_idx, path in enumerate(audio_paths)
            ]
        all_results = [outer_future.result() for outer_future in outer_futures]

    # # process each audio file
    # for path in audio_paths:
        # main_process(path)
