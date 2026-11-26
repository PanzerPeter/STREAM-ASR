❯ PYTHONPATH=. .venv/bin/python scripts/build_cv_manifests.py
14:10:40 │ INFO    │ cv-dev: 16,401 of 16,403 clips (unspellable dropped)
14:10:41 │ INFO    │ cv-test: 16,402 of 16,403 clips (unspellable dropped)
14:10:41 │ INFO    │ cv-train candidates: 1,335,422 clips; dropped {'heldout_clip': 32806, 'heldout_speaker': 4697, 'unspellable': 312, 'heldout_sentence': 556617}
14:11:01 │ INFO    │ decoded 50,000/1,335,422 clips (2455/s)
14:11:19 │ INFO    │ decoded 100,000/1,335,422 clips (2606/s)
14:11:37 │ INFO    │ decoded 150,000/1,335,422 clips (2646/s)
14:11:55 │ INFO    │ decoded 200,000/1,335,422 clips (2704/s)
14:12:12 │ INFO    │ decoded 250,000/1,335,422 clips (2721/s)
14:12:30 │ INFO    │ decoded 300,000/1,335,422 clips (2739/s)
14:12:47 │ INFO    │ decoded 350,000/1,335,422 clips (2760/s)
14:13:05 │ INFO    │ decoded 400,000/1,335,422 clips (2772/s)
14:13:22 │ INFO    │ decoded 450,000/1,335,422 clips (2786/s)
14:13:39 │ INFO    │ decoded 500,000/1,335,422 clips (2806/s)
14:13:55 │ INFO    │ decoded 550,000/1,335,422 clips (2822/s)
14:14:12 │ INFO    │ decoded 600,000/1,335,422 clips (2834/s)
14:14:29 │ INFO    │ decoded 650,000/1,335,422 clips (2849/s)
14:14:45 │ INFO    │ decoded 700,000/1,335,422 clips (2860/s)
14:15:02 │ INFO    │ decoded 750,000/1,335,422 clips (2874/s)
14:15:18 │ INFO    │ decoded 800,000/1,335,422 clips (2889/s)
14:15:33 │ INFO    │ decoded 850,000/1,335,422 clips (2903/s)
14:15:49 │ INFO    │ decoded 900,000/1,335,422 clips (2921/s)
14:16:05 │ INFO    │ decoded 950,000/1,335,422 clips (2926/s)
14:16:22 │ INFO    │ decoded 1,000,000/1,335,422 clips (2933/s)
14:16:38 │ INFO    │ decoded 1,050,000/1,335,422 clips (2937/s)
14:16:54 │ INFO    │ decoded 1,100,000/1,335,422 clips (2942/s)
14:17:12 │ INFO    │ decoded 1,150,000/1,335,422 clips (2942/s)
14:17:29 │ INFO    │ decoded 1,200,000/1,335,422 clips (2938/s)
14:17:46 │ INFO    │ decoded 1,250,000/1,335,422 clips (2936/s)
14:18:04 │ INFO    │ decoded 1,300,000/1,335,422 clips (2933/s)
14:18:18 │ INFO    │ decoded 1,335,422/1,335,422 clips (2922/s)
14:18:31 │ INFO    │ cv manifests written: BuildCommonVoiceManifestResponse(train_rows=1335344, dev_rows=16401, test_rows=16402, dev_subset_rows=2000, dropped={'heldout_clip': 32806, 'heldout_speaker': 4697, 'unspellable': 312, 'heldout_sentence': 556617, 'undecodable': 0, 'too_fast': 78}, trimmed_rows=773243, train_hours_untrimmed=2030.91887328125, train_hours=1934.5456672916666)
cv-train 1,335,344 clips, 2,030.9 h -> 1,934.5 h after trimming (773,243 clips trimmed)
cv-train dropped: {'heldout_clip': 32806, 'heldout_speaker': 4697, 'unspellable': 312, 'heldout_sentence': 556617, 'undecodable': 0, 'too_fast': 78}
cv-dev 16,401 · cv-test 16,402 · cv-dev2k 2,000

 󱄅  󰇘/STREAM-ASR   main !?  󱑎 14:18  
❯ PYTHONPATH=. .venv/bin/python scripts/write_trim_samples.py --out /tmp/trim-check --n 20
common_voice_en_23921863: cut 0.00 s head, 0.13 s tail | THE STATION HAS AN ORNATE ENTRANCE FACING THE RIVER
common_voice_en_24980617: cut 0.09 s head, 0.00 s tail | INTEGRONS CAN ALSO BE FOUND IN CHROMOSOMES
common_voice_en_18451117: cut 0.41 s head, 0.61 s tail | DO YOU WANT FRIES WITH THAT
common_voice_en_20947744: cut 0.25 s head, 0.00 s tail | I WALKED OFF OF WHAT'S NEW PUSSYCAT
common_voice_en_27203537: cut 0.00 s head, 0.28 s tail | IN THE FINAL HE FAINTED IN THE HEATWAVE AND WAS CARRIED AWAY BY MEDICS
common_voice_en_26953810: cut 0.00 s head, 0.09 s tail | THEY WILL TREAT ME WITH MORE RESPECT NOW I GUESS
common_voice_en_24498042: cut 0.00 s head, 0.08 s tail | JONES'S MOVE TO HARLEM WAS SHORT LIVED
common_voice_en_22048977: cut 0.00 s head, 0.35 s tail | FURTHER HEADLINE FESTIVAL SHOWS WERE ANNOUNCED IN THE COMING WEEKS
common_voice_en_26583735: cut 0.50 s head, 0.09 s tail | THESE VERTICAL SLABS OF STONE DEPICT TOMBSTONES RELIGIOUS USAGE AND BOUNDARIES
common_voice_en_23589894: cut 0.00 s head, 0.16 s tail | VALVES ARE MOST OFTEN PISTON STYLE
common_voice_en_29633111: cut 0.03 s head, 0.33 s tail | HE IS A VEGETARIAN AND A MACARTHUR FELLOW
common_voice_en_20300011: cut 0.50 s head, 0.00 s tail | THE WINNING REMIXES WERE INCLUDED ON THIS ALBUM
common_voice_en_27117583: cut 0.00 s head, 0.59 s tail | SUBZERO WAS WELL RECEIVED BY CRITICS
common_voice_en_19598194: cut 0.00 s head, 0.15 s tail | TOWNHOUSES AND HOMES ON ADJACENT STREETS AVERAGE SALES PRICES IN SEVEN FIGURES
common_voice_en_21505089: cut 0.07 s head, 0.12 s tail | THE PARISH HAS A CHURCH OF ENGLAND PRIMARY SCHOOL
common_voice_en_19603722: cut 0.00 s head, 0.67 s tail | IT HAD BETTER POLITICAL ORGANIZATION THAN ITS NEIGHBORS AND A SOMEWHAT FEUDAL SOCIAL STRUCTURE
common_voice_en_18940526: cut 0.00 s head, 0.17 s tail | AND I'M NOT GIVING THAT UP TO BE PRESIDENT OF THE UNITED STATES
common_voice_en_31937447: cut 0.19 s head, 0.06 s tail | ITS CURRENT AND FIRST MINISTER IS MYKHAILO FEDOROV
common_voice_en_20835267: cut 0.00 s head, 0.12 s tail | IN HIGH SCHOOL SHE WAS A MEMBER OF THE NATIONAL HONOR SOCIETY
common_voice_en_27482587: cut 0.13 s head, 0.00 s tail | DESPITE THIS JAGUARS STILL FETCH CONSIDERABLY LESS THAN TELECASTERS AND STRATOCASTERS OF SIMILAR VINTAGE
773,243 of the manifest's rows are trimmed; pairs written to /tmp/trim-check


stage             mode           WER     CER       RTF  finalize   partial      n   word errors
-----------------------------------------------------------------------------------------------
greedy_transducer offline     36.27%  19.23%     0.004    0.008s         -  16402   56926/156957
greedy_transducer streaming   43.96%  23.63%     0.035    0.007s      14ms  16402   68997/156957
beam              offline     35.92%  18.98%     0.020    0.107s         -  16402   56377/156957
beam              streaming   43.38%  23.27%     0.052    0.111s      13ms  16402   68094/156957
beam_lm           offline     33.25%  18.23%     0.020    0.111s         -  16402   52183/156957
beam_lm           streaming   40.40%  22.40%     0.054    0.119s      13ms  16402   63404/156957

stage             mode             first-2        middle        last-2    emit p50/p90
--------------------------------------------------------------------------------------
greedy_transducer offline       28.6 (1.7)    29.5 (3.6)    37.3 (6.7)               -
greedy_transducer streaming     38.7 (2.2)    35.1 (4.3)    40.6 (8.1)      0.08/0.32s
beam              offline       27.8 (1.6)    28.7 (3.2)    36.4 (5.6)               -
beam              streaming     37.8 (2.0)    34.0 (4.0)    39.3 (7.0)      0.08/0.32s
beam_lm           offline       26.4 (1.6)    27.0 (3.8)    33.9 (7.2)               -
beam_lm           streaming     35.6 (1.9)    32.3 (4.8)    36.7 (8.5)      0.08/0.32s

wrote runs/eval/report-cv.json  (total 15222s)

❯ mv runs/eval/report-cv.json runs/eval/report-cv-v2.0.json

 󱄅  󰇘/STREAM-ASR   main !?  󱑎 18:42  
❯ mkdir -p data/v2.0

 󱄅  󰇘/STREAM-ASR   main !?  󱑎 18:42  
❯ mv data/tokenizer data/features/cmvn.pt data/lm_data data/checkpoints/lm_best.pt \
  data/checkpoints/transducer_avg.pt data/v2.0/

 󱄅  󰇘/STREAM-ASR   main !?  󱑎 18:42  
❯ PYTHONPATH=. .venv/bin/python scripts/train_tokenizer.py
sentencepiece_trainer.cc(78) LOG(INFO) Starts training with : 
trainer_spec {
  input: /tmp/tmp9t_wr43y.txt
  input_format: 
  model_prefix: data/tokenizer/bpe500
  model_type: UNIGRAM
  vocab_size: 500
  self_test_sample_size: 0
  character_coverage: 1
  input_sentence_size: 0
  shuffle_input_sentence: 1
  seed_sentencepiece_size: 1000000
  shrinking_factor: 0.75
  max_sentence_length: 4192
  num_threads: 16
  num_sub_iterations: 2
  max_sentencepiece_length: 16
  split_by_unicode_script: 1
  split_by_number: 1
  split_by_whitespace: 1
  split_digits: 0
  pretokenization_delimiter: 
  treat_whitespace_as_suffix: 0
  allow_whitespace_only_pieces: 0
  required_chars: 
  byte_fallback: 0
  vocabulary_output_piece_score: 1
  train_extremely_large_corpus: 0
  seed_sentencepieces_file: 
  hard_vocab_limit: 1
  use_all_vocab: 0
  unk_id: 0
  bos_id: -1
  eos_id: -1
  pad_id: -1
  unk_piece: <unk>
  bos_piece: <s>
  eos_piece: </s>
  pad_piece: <pad>
  unk_surface:  ⁇ 
  enable_differential_privacy: 0
  differential_privacy_noise_level: 0
  differential_privacy_clipping_threshold: 0
}
normalizer_spec {
  name: nmt_nfkc
  add_dummy_prefix: 1
  remove_extra_whitespaces: 1
  escape_whitespaces: 1
  normalization_rule_tsv: 
}
denormalizer_spec {}
trainer_interface.cc(355) LOG(INFO) SentenceIterator is not specified. Using MultiFileSentenceIterator.
trainer_interface.cc(186) LOG(INFO) Loading corpus: /tmp/tmp9t_wr43y.txt
trainer_interface.cc(148) LOG(INFO) Loaded 1000000 lines
trainer_interface.cc(125) LOG(WARNING) Too many sentences are loaded! (1616585), which may slow down training.
trainer_interface.cc(127) LOG(WARNING) Consider using --input_sentence_size=<size> and --shuffle_input_sentence=true.
trainer_interface.cc(130) LOG(WARNING) They allow to randomly sample <size> sentences from the entire corpus.
trainer_interface.cc(411) LOG(INFO) Loaded all 1616585 sentences
trainer_interface.cc(427) LOG(INFO) Adding meta_piece: <unk>
trainer_interface.cc(432) LOG(INFO) Normalizing sentences...
trainer_interface.cc(541) LOG(INFO) all chars count=129374683
trainer_interface.cc(562) LOG(INFO) Alphabet size=28
trainer_interface.cc(563) LOG(INFO) Final character coverage=1
trainer_interface.cc(594) LOG(INFO) Done! preprocessed 1616585 sentences.
unigram_model_trainer.cc(265) LOG(INFO) Making suffix array...
unigram_model_trainer.cc(269) LOG(INFO) Extracting frequent sub strings... node_num=67789284
unigram_model_trainer.cc(312) LOG(INFO) Initialized 662089 seed sentencepieces
trainer_interface.cc(600) LOG(INFO) Tokenizing input sentences with whitespace: 1616585
trainer_interface.cc(611) LOG(INFO) Done! 296601
unigram_model_trainer.cc(602) LOG(INFO) Using 296601 sentences for EM training
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=207790 obj=9.81759 num_tokens=544030 num_tokens/piece=2.61817
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=166111 obj=7.78489 num_tokens=539303 num_tokens/piece=3.24664
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=124581 obj=7.73211 num_tokens=569150 num_tokens/piece=4.56851
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=124487 obj=7.72623 num_tokens=569107 num_tokens/piece=4.57162
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=93363 obj=7.75209 num_tokens=616067 num_tokens/piece=6.59862
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=93362 obj=7.74635 num_tokens=616009 num_tokens/piece=6.59807
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=70021 obj=7.78607 num_tokens=665219 num_tokens/piece=9.50028
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=70021 obj=7.7787 num_tokens=665136 num_tokens/piece=9.49909
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=52515 obj=7.83363 num_tokens=712212 num_tokens/piece=13.5621
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=52515 obj=7.82363 num_tokens=712155 num_tokens/piece=13.561
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=39386 obj=7.89876 num_tokens=757324 num_tokens/piece=19.2283
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=39386 obj=7.8848 num_tokens=757309 num_tokens/piece=19.2279
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=29539 obj=7.98536 num_tokens=800999 num_tokens/piece=27.1167
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=29539 obj=7.96597 num_tokens=801045 num_tokens/piece=27.1182
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=22154 obj=8.09548 num_tokens=844769 num_tokens/piece=38.1317
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=22154 obj=8.06986 num_tokens=844951 num_tokens/piece=38.1399
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=16615 obj=8.22861 num_tokens=888636 num_tokens/piece=53.484
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=16615 obj=8.19638 num_tokens=888772 num_tokens/piece=53.4921
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=12461 obj=8.38979 num_tokens=931820 num_tokens/piece=74.7789
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=12461 obj=8.35016 num_tokens=932051 num_tokens/piece=74.7974
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=9345 obj=8.5761 num_tokens=976575 num_tokens/piece=104.502
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=9345 obj=8.52981 num_tokens=976794 num_tokens/piece=104.526
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=7008 obj=8.78976 num_tokens=1026184 num_tokens/piece=146.43
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=7008 obj=8.7354 num_tokens=1026370 num_tokens/piece=146.457
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=5256 obj=9.03469 num_tokens=1074059 num_tokens/piece=204.349
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=5256 obj=8.97063 num_tokens=1074478 num_tokens/piece=204.429
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=3942 obj=9.31953 num_tokens=1120687 num_tokens/piece=284.294
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=3942 obj=9.2478 num_tokens=1120941 num_tokens/piece=284.358
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=2956 obj=9.62645 num_tokens=1163815 num_tokens/piece=393.713
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=2956 obj=9.54898 num_tokens=1163853 num_tokens/piece=393.726
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=2217 obj=9.967 num_tokens=1213372 num_tokens/piece=547.304
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=2217 obj=9.88123 num_tokens=1213666 num_tokens/piece=547.436
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=1662 obj=10.3351 num_tokens=1259366 num_tokens/piece=757.741
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=1662 obj=10.2398 num_tokens=1259379 num_tokens/piece=757.749
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=1246 obj=10.7152 num_tokens=1309085 num_tokens/piece=1050.63
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=1246 obj=10.6126 num_tokens=1309053 num_tokens/piece=1050.6
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=934 obj=11.1189 num_tokens=1364594 num_tokens/piece=1461.02
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=934 obj=11.0052 num_tokens=1364603 num_tokens/piece=1461.03
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=700 obj=11.5142 num_tokens=1418381 num_tokens/piece=2026.26
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=700 obj=11.3993 num_tokens=1418461 num_tokens/piece=2026.37
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=550 obj=11.8353 num_tokens=1467682 num_tokens/piece=2668.51
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=550 obj=11.7241 num_tokens=1467766 num_tokens/piece=2668.67
trainer_interface.cc(689) LOG(INFO) Saving model: data/tokenizer/bpe500.model
trainer_interface.cc(701) LOG(INFO) Saving vocabs: data/tokenizer/bpe500.vocab
tokenizer: data/tokenizer/bpe500.model

19:08:43 │ INFO    │ cv-train: 1330000/1335344 (100%) 1408 utt/s ETA 0m
19:08:45 │ INFO    │ cv-train: 1332000/1335344 (100%) 1408 utt/s ETA 0m
19:08:46 │ INFO    │ cv-train: 1334000/1335344 (100%) 1408 utt/s ETA 0m
19:08:47 │ INFO    │ cv-train: 1335344/1335344 (100%) 1408 utt/s ETA 0m
cv-train: 1335344 utts cached in data/features/mel
19:08:47 │ INFO    │ dev: 2703 utts already cached for this manifest, skipping
dev: 2703 utts cached in data/features/mel
19:08:47 │ INFO    │ dev-other: 2864 utts already cached for this manifest, skipping
dev-other: 2864 utts cached in data/features/mel
19:08:50 │ INFO    │ cv-dev2k: 2000/2000 (100%) 724 utt/s ETA 0m
cv-dev2k: 2000 utts cached in data/features/mel
19:08:50 │ INFO    │ test: 2620 utts already cached for this manifest, skipping
test: 2620 utts cached in data/features/mel
19:08:50 │ INFO    │ test-other: 2939 utts already cached for this manifest, skipping
test-other: 2939 utts cached in data/features/mel

❯ PYTHONPATH=. .venv/bin/python scripts/precompute_features.py
19:17:06 │ INFO    │ train_sp2: 562482 utts already cached for this manifest, skipping
train_sp2: 562482 utts cached in data/features/mel
19:17:08 │ INFO    │ cv-train: 1335344 utts already cached for this manifest, skipping
cv-train: 1335344 utts cached in data/features/mel
19:17:08 │ INFO    │ dev: 2703 utts already cached for this manifest, skipping
dev: 2703 utts cached in data/features/mel
19:17:08 │ INFO    │ dev-other: 2864 utts already cached for this manifest, skipping
dev-other: 2864 utts cached in data/features/mel
19:17:08 │ INFO    │ cv-dev2k: 2000 utts already cached for this manifest, skipping
cv-dev2k: 2000 utts cached in data/features/mel
19:17:08 │ INFO    │ test: 2620 utts already cached for this manifest, skipping
test: 2620 utts cached in data/features/mel
19:17:08 │ INFO    │ test-other: 2939 utts already cached for this manifest, skipping
test-other: 2939 utts cached in data/features/mel

19:43:06 │ INFO    │ lines   55,423,225 │ train   979,787,443 words / 2,232,008,324 tokens (  4464.0 MB) │   110589 lines/s
19:43:08 │ INFO    │ lines   55,627,266 │ train   983,341,627 words / 2,240,008,332 tokens (  4480.0 MB) │   110546 lines/s
19:43:10 │ INFO    │ lines   55,832,435 │ train   986,885,629 words / 2,248,008,355 tokens (  4496.0 MB) │   110556 lines/s
19:43:11 │ INFO    │ lines   56,032,221 │ train   990,337,256 words / 2,256,008,379 tokens (  4512.0 MB) │   110561 lines/s
19:43:13 │ INFO    │ lines   56,227,521 │ train   993,787,364 words / 2,264,008,441 tokens (  4528.0 MB) │   110559 lines/s
19:43:15 │ INFO    │ lines   56,425,848 │ train   997,253,396 words / 2,272,008,448 tokens (  4544.0 MB) │   110561 lines/s
19:43:16 │ INFO    │ Done in 512s: 56,583,193 lines │ train 2,278,389,504 tokens -> data/lm_data/train.bin │ val 465,020 tokens -> data/lm_data/val.bin

20:49:30 │ INFO    │ step  57,750/60,000 │ loss  0.799 (rnnt  0.42 ctc  0.77) │ chunk 0 left 128 │ lr 6.54e-05 │ 12.76 it/s │ eta 0:02:56
20:49:48 │ INFO    │ step  58,000/60,000 │ loss  1.518 (rnnt  0.82 ctc  1.39) │ chunk 16 left 128 │ lr 5.88e-05 │ 13.48 it/s │ eta 0:02:28
20:50:08 │ INFO    │ step  58,250/60,000 │ loss  0.979 (rnnt  0.52 ctc  0.91) │ chunk 8 left 128 │ lr 5.22e-05 │ 12.96 it/s │ eta 0:02:15
20:50:26 │ INFO    │ step  58,500/60,000 │ loss  1.127 (rnnt  0.67 ctc  0.91) │ chunk 16 left -1 │ lr 4.56e-05 │ 13.47 it/s │ eta 0:01:51
20:50:45 │ INFO    │ step  58,750/60,000 │ loss  0.935 (rnnt  0.49 ctc  0.90) │ chunk 8 left -1 │ lr 3.91e-05 │ 13.32 it/s │ eta 0:01:33
20:51:03 │ INFO    │ step  59,000/60,000 │ loss  1.232 (rnnt  0.54 ctc  1.39) │ chunk 8 left 64 │ lr 3.27e-05 │ 13.47 it/s │ eta 0:01:14
20:51:22 │ INFO    │ step  59,250/60,000 │ loss  0.920 (rnnt  0.47 ctc  0.90) │ chunk 16 left -1 │ lr 2.63e-05 │ 13.47 it/s │ eta 0:00:55
20:51:42 │ INFO    │ step  59,500/60,000 │ loss  1.014 (rnnt  0.55 ctc  0.93) │ chunk 16 left 128 │ lr 2.00e-05 │ 12.45 it/s │ eta 0:00:40
20:52:01 │ INFO    │ step  59,750/60,000 │ loss  0.670 (rnnt  0.36 ctc  0.61) │ chunk 0 left -1 │ lr 1.37e-05 │ 13.48 it/s │ eta 0:00:18
20:52:43 │ SUCCESS │ ctc_wer_clean 0.0933 │ ctc_wer_stream_clean 0.1178 │ transducer_wer_clean 0.0686 │ transducer_wer_stream_clean 0.0878 │ blank_frac_clean 0.4568 │ ctc_wer_stream_gap_clean 0.0245 │ ctc_wer_cv 0.2938 │ ctc_wer_stream_cv 0.3458 │ transducer_wer_cv 0.2474 │ transducer_wer_stream_cv 0.3023 │ blank_frac_cv 0.6632 │ ctc_wer_stream_gap_cv 0.0520 │ ctc_wer_other 0.1855 │ ctc_wer_stream_other 0.2284 │ transducer_wer_other 0.1500 │ transducer_wer_stream_other 0.1881 │ blank_frac_other 0.4895 │ ctc_wer_stream_gap_other 0.0430 │ band_loss_gap 0.0000 │ gate_sat_frac 0.0946 @ step 60,000  ← best
20:52:43 │ SUCCESS │ done │ steps 60,000 · 1:33:30 · 0.094 s/step this process · peak 4.24 GiB · best transducer_wer_stream_other 0.1881

❯ sed -i 's/^stack_dims: \[256, 384, 512, 384\]/stack_dims: [320, 512, 640, 512]/' config/model.yaml
PYTHONPATH=. .venv/bin/python scripts/profile_transducer_step.py
.venv/bin/python -m src.slices.TrainAcousticModel.train_transducer --fresh --total-steps 60000 \
  --ckpt-dir data/checkpoints/proxy-wide --log-dir runs/proxy-wide
  63.08 M params · compiled True
  loader supplies 6493x realtime of distinct audio
W1121 21:21:33.082000 44673 .venv/lib/python3.12/site-packages/torch/_inductor/utils.py:1731] [0/0_1] Not enough SMs to use max_autotune_gemm mode
  band off │  147.0 ms/step │   1881x realtime │ peak 4.55 GiB
  band on  │  136.2 ms/step │   2034x realtime │ peak 3.59 GiB
/home/peter/Documents/Codes/AI/STREAM-ASR/.venv/lib/python3.12/site-packages/torch/profiler/profiler.py:224: UserWarning: Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True.
  _warn_once(

  6324 kernel launches/step (band on)
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                               Optimizer.step#Muon.step         0.00%       0.000us         0.00%       0.000us       0.000us     207.611ms        38.83%     207.611ms      51.903ms             4  
                                               aten::mm         1.24%       7.783ms         2.73%      17.234ms      16.571us     120.354ms        22.51%     124.024ms     119.254us          1040  
                                              aten::bmm         0.56%       3.549ms         0.75%       4.733ms       7.889us     111.033ms        20.77%     111.033ms     185.056us           600  
                             aten::convolution_backward         1.29%       8.131ms         1.82%      11.462ms     119.391us      70.603ms        13.21%      83.019ms     864.782us            96  
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us      50.539ms         9.45%      50.539ms     252.693us           200  
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us      41.221ms         7.71%      41.221ms     343.505us           120  
                                              aten::add         0.83%       5.211ms         1.41%       8.869ms       4.564us      39.999ms         7.48%      40.005ms      20.589us          1943  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      34.505ms         6.45%      34.505ms      52.599us           656  
void at::native::(anonymous namespace)::conv_depthwi...         0.00%       0.000us         0.00%       0.000us       0.000us      33.097ms         6.19%      33.097ms     376.104us            88  
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s1681...         0.00%       0.000us         0.00%       0.000us       0.000us      31.718ms         5.93%      31.718ms     149.614us           212  
                                              aten::mul         0.34%       2.144ms         0.63%       3.971ms       5.909us      31.604ms         5.91%      31.609ms      47.037us           672  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      25.829ms         4.83%      25.829ms      74.009us           349  
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s1681...         0.00%       0.000us         0.00%       0.000us       0.000us      22.255ms         4.16%      22.255ms     168.602us           132  
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s1681...         0.00%       0.000us         0.00%       0.000us       0.000us      21.617ms         4.04%      21.617ms     168.882us           128  
                                            aten::copy_         0.34%       2.124ms        68.89%     434.106ms     602.924us      21.573ms         4.04%      22.153ms      30.768us           720  
                                aten::_conv_depthwise2d         0.05%     296.434us         0.32%       2.019ms      22.938us      18.005ms         3.37%      19.698ms     223.845us            88  
void at::native::(anonymous namespace)::conv_depthwi...         0.00%       0.000us         0.00%       0.000us       0.000us      17.216ms         3.22%      17.216ms     204.952us            84  
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us      15.918ms         2.98%      15.918ms     397.952us            40  
void at::native::(anonymous namespace)::conv_depthwi...         0.00%       0.000us         0.00%       0.000us       0.000us      15.775ms         2.95%      15.775ms     187.792us            84  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      15.651ms         2.93%      15.651ms      79.852us           196  
                    aten::_efficient_attention_backward         0.09%     579.945us         0.28%       1.795ms      22.438us      14.031ms         2.62%      14.281ms     178.516us            80  
fmha_cutlassB_bf16_aligned_64x64_k64_sm80(PyTorchMem...         0.00%       0.000us         0.00%       0.000us       0.000us      14.031ms         2.62%      14.031ms     175.391us            80  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      13.226ms         2.47%      13.226ms       8.753us          1511  
                                    aten::_foreach_add_         0.06%     365.601us         0.09%     538.792us      10.361us       9.848ms         1.84%       9.848ms     189.385us            52  
void at::native::(anonymous namespace)::multi_tensor...         0.00%       0.000us         0.00%       0.000us       0.000us       9.837ms         1.84%       9.837ms     153.701us            64  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 630.145ms
Self CUDA time total: 534.652ms



23:07:59 │ INFO    │ step  58,250/60,000 │ loss  0.896 (rnnt  0.47 ctc  0.86) │ chunk 8 left 128 │ lr 5.22e-05 │  9.75 it/s │ eta 0:02:59
23:08:25 │ INFO    │ step  58,500/60,000 │ loss  1.075 (rnnt  0.65 ctc  0.85) │ chunk 16 left -1 │ lr 4.56e-05 │  9.77 it/s │ eta 0:02:33
23:08:50 │ INFO    │ step  58,750/60,000 │ loss  0.834 (rnnt  0.43 ctc  0.80) │ chunk 8 left -1 │ lr 3.91e-05 │  9.80 it/s │ eta 0:02:07
23:09:17 │ INFO    │ step  59,000/60,000 │ loss  1.177 (rnnt  0.51 ctc  1.34) │ chunk 8 left 64 │ lr 3.27e-05 │  9.44 it/s │ eta 0:01:45
23:09:42 │ INFO    │ step  59,250/60,000 │ loss  0.857 (rnnt  0.45 ctc  0.82) │ chunk 16 left -1 │ lr 2.63e-05 │  9.76 it/s │ eta 0:01:16
23:10:09 │ INFO    │ step  59,500/60,000 │ loss  0.961 (rnnt  0.54 ctc  0.84) │ chunk 16 left 128 │ lr 2.00e-05 │  9.42 it/s │ eta 0:00:53
23:10:35 │ INFO    │ step  59,750/60,000 │ loss  0.626 (rnnt  0.34 ctc  0.57) │ chunk 0 left -1 │ lr 1.37e-05 │  9.70 it/s │ eta 0:00:25
23:11:29 │ SUCCESS │ ctc_wer_clean 0.0836 │ ctc_wer_stream_clean 0.1068 │ transducer_wer_clean 0.0628 │ transducer_wer_stream_clean 0.0808 │ blank_frac_clean 0.4514 │ ctc_wer_stream_gap_clean 0.0232 │ ctc_wer_cv 0.2724 │ ctc_wer_stream_cv 0.3284 │ transducer_wer_cv 0.2349 │ transducer_wer_stream_cv 0.2865 │ blank_frac_cv 0.6603 │ ctc_wer_stream_gap_cv 0.0560 │ ctc_wer_other 0.1707 │ ctc_wer_stream_other 0.2102 │ transducer_wer_other 0.1387 │ transducer_wer_stream_other 0.1736 │ blank_frac_other 0.4861 │ ctc_wer_stream_gap_other 0.0396 │ band_loss_gap 0.0000 │ gate_sat_frac 0.1099 @ step 60,000  ← best
23:11:29 │ SUCCESS │ done │ steps 60,000 · 1:49:05 · 0.109 s/step this process · peak 5.07 GiB · best transducer_wer_stream_other 0.1736

04:39:21 │ INFO    │ step  69,550/70,000 (99.4%) │ loss   2.205 │ lr 1.08e-07 │  3.82 it/s │ eta 0:01:57
04:39:35 │ INFO    │ step  69,600/70,000 (99.4%) │ loss   2.203 │ lr 8.54e-08 │  3.83 it/s │ eta 0:01:44
04:39:48 │ INFO    │ step  69,650/70,000 (99.5%) │ loss   2.195 │ lr 6.54e-08 │  3.83 it/s │ eta 0:01:31
04:40:01 │ INFO    │ step  69,700/70,000 (99.6%) │ loss   2.205 │ lr 4.80e-08 │  3.82 it/s │ eta 0:01:18
04:40:14 │ INFO    │ step  69,750/70,000 (99.6%) │ loss   2.202 │ lr 3.34e-08 │  3.82 it/s │ eta 0:01:05
04:40:27 │ INFO    │ step  69,800/70,000 (99.7%) │ loss   2.207 │ lr 2.13e-08 │  3.82 it/s │ eta 0:00:52
04:40:40 │ INFO    │ step  69,850/70,000 (99.8%) │ loss   2.197 │ lr 1.20e-08 │  3.82 it/s │ eta 0:00:39
04:40:53 │ INFO    │ step  69,900/70,000 (99.9%) │ loss   2.199 │ lr 5.34e-09 │  3.83 it/s │ eta 0:00:26
04:41:06 │ INFO    │ step  69,950/70,000 (99.9%) │ loss   2.198 │ lr 1.33e-09 │  3.82 it/s │ eta 0:00:13
04:41:19 │ INFO    │ step  70,000/70,000 (100.0%) │ loss   2.197 │ lr 0.00e+00 │  3.83 it/s │ eta 0:00:00
04:41:21 │ INFO    │ val ppl   26.903  (best 26.022)  @ step 70,000
04:41:22 │ SUCCESS │ STREAM-LM complete │ steps 70,000 · elapsed 5:10:24 · best val ppl 26.022
04:41:22 │ SUCCESS │ last → data/checkpoints/lm_last.pt · best → data/checkpoints/lm_best.pt
STREAM-LM done. best val perplexity = 26.022 -> data/checkpoints/lm_best.pt


21:27:18 │ INFO    │ step 1,199,750/1,200,000 │ loss  0.215 (rnnt  0.10 ctc  0.23) │ chunk 16 left 128 │ lr 7.81e-06 │  9.63 it/s │ eta 0:00:25
21:28:15 │ SUCCESS │ ctc_wer_clean 0.0318 │ ctc_wer_stream_clean 0.0478 │ transducer_wer_clean 0.0262 │ transducer_wer_stream_clean 0.0349 │ blank_frac_clean 0.4675 │ ctc_wer_stream_gap_clean 0.0160 │ ctc_wer_cv 0.1450 │ ctc_wer_stream_cv 0.1869 │ transducer_wer_cv 0.1231 │ transducer_wer_stream_cv 0.1617 │ blank_frac_cv 0.6545 │ ctc_wer_stream_gap_cv 0.0418 │ ctc_wer_other 0.0799 │ ctc_wer_stream_other 0.1128 │ transducer_wer_other 0.0678 │ transducer_wer_stream_other 0.0912 │ blank_frac_other 0.4906 │ ctc_wer_stream_gap_other 0.0329 │ band_loss_gap 0.0000 │ gate_sat_frac 0.0152 @ step 1,200,000  ← best
21:28:15 │ SUCCESS │ done │ steps 1,200,000 · 1 day, 11:39:17 · 0.107 s/step this process · peak 5.07 GiB · best transducer_wer_stream_other 0.0912

❯ PYTHONPATH=. .venv/bin/python scripts/average_checkpoints.py --last-n 5
averaged 5 checkpoints -> data/checkpoints/transducer_avg.pt
  data/checkpoints/transducer_step1177200.pt
  data/checkpoints/transducer_step1182600.pt
  data/checkpoints/transducer_step1188000.pt
  data/checkpoints/transducer_step1193400.pt
  data/checkpoints/transducer_step1198800.pt

20:12:52 │ INFO    │ step 599,750/600,000 │ loss  0.286 (rnnt  0.15 ctc  0.27) │ chunk 16 left 64 │ lr 8.12e-06 │  8.79 it/s │ eta 0:00:28
20:13:51 │ SUCCESS │ ctc_wer_clean 0.0354 │ ctc_wer_stream_clean 0.0528 │ transducer_wer_clean 0.0294 │ transducer_wer_stream_clean 0.0392 │ blank_frac_clean 0.4651 │ ctc_wer_stream_gap_clean 0.0174 │ ctc_wer_cv 0.1538 │ ctc_wer_stream_cv 0.2018 │ transducer_wer_cv 0.1315 │ transducer_wer_stream_cv 0.1757 │ blank_frac_cv 0.6558 │ ctc_wer_stream_gap_cv 0.0480 │ ctc_wer_other 0.0877 │ ctc_wer_stream_other 0.1238 │ transducer_wer_other 0.0739 │ transducer_wer_stream_other 0.1003 │ blank_frac_other 0.4904 │ ctc_wer_stream_gap_other 0.0361 │ band_loss_gap 0.0000 │ gate_sat_frac 0.0351 @ step 600,000  ← best
20:13:51 │ SUCCESS │ done │ steps 600,000 · 4:55:28 · 0.118 s/step this process · peak 4.12 GiB · best transducer_wer_stream_other 0.1003
averaged 5 checkpoints -> data/checkpoints/b600k/transducer_avg.pt
  data/checkpoints/b600k/transducer_step577800.pt
  data/checkpoints/b600k/transducer_step583200.pt
  data/checkpoints/b600k/transducer_step588600.pt
  data/checkpoints/b600k/transducer_step594000.pt
  data/checkpoints/b600k/transducer_step599400.pt

  trainer_interface.cc(148) LOG(INFO) Loaded 101000000 lines
trainer_interface.cc(148) LOG(INFO) Loaded 102000000 lines
trainer_interface.cc(125) LOG(WARNING) Too many sentences are loaded! (2000000), which may slow down training.
trainer_interface.cc(127) LOG(WARNING) Consider using --input_sentence_size=<size> and --shuffle_input_sentence=true.
trainer_interface.cc(130) LOG(WARNING) They allow to randomly sample <size> sentences from the entire corpus.
trainer_interface.cc(413) LOG(INFO) Sampled 2000000 sentences from 102469463 sentences.
trainer_interface.cc(418) LOG(INFO) Skipped 45 too long sentences.
trainer_interface.cc(427) LOG(INFO) Adding meta_piece: <unk>
trainer_interface.cc(427) LOG(INFO) Adding meta_piece: </s>
trainer_interface.cc(432) LOG(INFO) Normalizing sentences...
trainer_interface.cc(541) LOG(INFO) all chars count=201945759
trainer_interface.cc(562) LOG(INFO) Alphabet size=28
trainer_interface.cc(563) LOG(INFO) Final character coverage=1
trainer_interface.cc(594) LOG(INFO) Done! preprocessed 2000000 sentences.
unigram_model_trainer.cc(265) LOG(INFO) Making suffix array...
unigram_model_trainer.cc(269) LOG(INFO) Extracting frequent sub strings... node_num=103852006
unigram_model_trainer.cc(312) LOG(INFO) Initialized 765505 seed sentencepieces
trainer_interface.cc(600) LOG(INFO) Tokenizing input sentences with whitespace: 2000000
trainer_interface.cc(611) LOG(INFO) Done! 374205
unigram_model_trainer.cc(602) LOG(INFO) Using 374205 sentences for EM training
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=233288 obj=9.79805 num_tokens=722498 num_tokens/piece=3.09702
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=187763 obj=7.72486 num_tokens=710592 num_tokens/piece=3.78452
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=140819 obj=7.67344 num_tokens=740589 num_tokens/piece=5.25916
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=140707 obj=7.66928 num_tokens=740434 num_tokens/piece=5.26224
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=105530 obj=7.68652 num_tokens=795394 num_tokens/piece=7.53714
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=105528 obj=7.68303 num_tokens=795322 num_tokens/piece=7.5366
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=79146 obj=7.71089 num_tokens=854099 num_tokens/piece=10.7914
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=79146 obj=7.7055 num_tokens=854018 num_tokens/piece=10.7904
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=59359 obj=7.74624 num_tokens=911972 num_tokens/piece=15.3637
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=59359 obj=7.73858 num_tokens=911896 num_tokens/piece=15.3624
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=44519 obj=7.79701 num_tokens=969795 num_tokens/piece=21.7838
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=44519 obj=7.78592 num_tokens=969775 num_tokens/piece=21.7834
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=33389 obj=7.86762 num_tokens=1026217 num_tokens/piece=30.7352
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=33389 obj=7.85174 num_tokens=1026226 num_tokens/piece=30.7355
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=25041 obj=7.95911 num_tokens=1084708 num_tokens/piece=43.3173
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=25041 obj=7.93736 num_tokens=1084864 num_tokens/piece=43.3235
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=18780 obj=8.07547 num_tokens=1142549 num_tokens/piece=60.8386
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=18780 obj=8.04706 num_tokens=1142835 num_tokens/piece=60.8538
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=14085 obj=8.21647 num_tokens=1200132 num_tokens/piece=85.2064
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=14085 obj=8.18053 num_tokens=1200425 num_tokens/piece=85.2272
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=10563 obj=8.38325 num_tokens=1257220 num_tokens/piece=119.021
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=10563 obj=8.34112 num_tokens=1257644 num_tokens/piece=119.061
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=0 size=9011 obj=8.46255 num_tokens=1290840 num_tokens/piece=143.252
unigram_model_trainer.cc(618) LOG(INFO) EM sub_iter=1 size=9011 obj=8.43823 num_tokens=1290988 num_tokens/piece=143.268
trainer_interface.cc(689) LOG(INFO) Saving model: data/tokenizer/lm8k.model
trainer_interface.cc(701) LOG(INFO) Saving vocabs: data/tokenizer/lm8k.vocab
LM tokenizer: data/tokenizer/lm8k.model

16:17:11 │ INFO    │ lines  102,268,607 │ train 1,811,616,760 words / 2,320,004,362 tokens (  4640.0 MB) │   136472 lines/s
16:17:12 │ INFO    │ Done in 751s: 102,469,508 lines │ train 2,324,198,514 tokens -> data/lm_data/train.bin │ val 258,904 tokens -> data/lm_data/val.bin

23:34:53 │ INFO    │ step 139,850/140,000 (99.9%) │ ce   3.849 │ lr 2.15e-06 │  5.46 it/s │ eta 0:00:27
23:35:02 │ INFO    │ step 139,900/140,000 (99.9%) │ ce   3.844 │ lr 1.43e-06 │  5.46 it/s │ eta 0:00:18
23:35:11 │ INFO    │ step 139,950/140,000 (100.0%) │ ce   3.839 │ lr 7.15e-07 │  5.46 it/s │ eta 0:00:09
23:35:20 │ INFO    │ step 140,000/140,000 (100.0%) │ ce   3.839 │ lr 0.00e+00 │  5.46 it/s │ eta 0:00:00
23:35:22 │ SUCCESS │ val ppl   52.402  ← best  @ step 140,000
23:35:23 │ SUCCESS │ STREAM-LM complete │ steps 140,000 · elapsed 7:14:55 · best val ppl 52.402
23:35:23 │ SUCCESS │ last → data/checkpoints/lm_last.pt · best → data/checkpoints/lm_best.pt
STREAM-LM done. best val perplexity = 52.402 -> data/checkpoints/lm_best.pt

stage             mode           WER     CER       RTF  finalize   partial      n   word errors
-----------------------------------------------------------------------------------------------
greedy_transducer offline     15.69%   7.84%     0.003    0.007s         -  16402   24622/156957
greedy_transducer streaming   20.07%  10.17%     0.031    0.006s      12ms  16402   31497/156957
beam              offline     15.43%   7.65%     0.016    0.089s         -  16402   24220/156957
beam              streaming   19.31%   9.73%     0.045    0.095s      12ms  16402   30315/156957
beam_lm           offline     13.42%   6.97%     0.017    0.093s         -  16402   21062/156957
beam_lm           streaming   16.80%   8.99%     0.046    0.099s      12ms  16402   26366/156957

stage             mode             first-2        middle        last-2    emit p50/p90
--------------------------------------------------------------------------------------
greedy_transducer offline       13.7 (1.7)    13.3 (2.6)    18.3 (3.9)               -
greedy_transducer streaming     20.5 (2.5)    16.8 (3.4)    20.4 (4.8)      0.08/0.32s
beam              offline       13.4 (1.4)    12.9 (2.2)    17.5 (3.3)               -
beam              streaming     19.4 (2.1)    16.0 (3.0)    19.4 (4.2)      0.08/0.32s
beam_lm           offline       12.1 (1.4)    11.2 (2.4)    14.9 (3.6)               -
beam_lm           streaming     16.9 (2.1)    14.0 (3.2)    17.0 (4.9)      0.08/0.32s

wrote runs/eval/report-cv.json  (total 12772s)