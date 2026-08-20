# Analysis-data dictionary

The public analysis begins from the two tidied datasets below. The RDS and CSV
versions are generated from the same R object. Acquisition files and participant
voice recordings are not released.

## Experiment 1

`experiment_1_analysis.rds` and `experiment_1_analysis.csv` contain 11,200 trials from 56
participants before the exclusions applied in the analysis report.

| Variable | Meaning |
| --- | --- |
| `SubjID` | Pseudonymous participant number |
| `subject_age` | Age in years |
| `subject_gender` | Self-reported gender |
| `familiar` | Auditory prior indicator: 0 = weak, 1 = strong |
| `interference` | Auditory perturbation: `noise` or `music` |
| `avg_block_music_vivid` | Block-level music-imagery vividness, 0-5 |
| `volume` | Signal strength used on the trial |
| `binary_resp` | Detection response: 0 = absent, 1 = present |
| `response_time` | Recorded response time |
| `trialID` | Trial number within participant-condition sequence |
| `lshs_score` | Launay-Slade Hallucination Scale revised total |
| `hittoolow` | Participant-level zero-hit exclusion flag |
| `avg_musicvividlessthan2` | Participant-level low-imagery exclusion flag |
| `response_Speech_vividness` | Signal vividness rating, 1-5 |
| `music_identified` | Derived recognition indicator; source free text is removed |

## Experiment 2

`experiment_2_analysis.rds` and `experiment_2_analysis.csv` contain 44,640 trials from 93
participants before the exclusions applied in the analysis report.

| Variable | Meaning |
| --- | --- |
| `subject_nr` | Pseudonymous participant number |
| `subject_age` | Age in years |
| `subject_gender` | Self-reported gender |
| `speech` | Voice condition: `self` or `distorted` |
| `interference` | Motor perturbation: none, foot-tapping, or articulatory suppression |
| `lshs_score` | Launay-Slade Hallucination Scale revised total |
| `volume` | Signal strength used on the trial |
| `resp_binary` | Detection response: 0 = absent, 1 = present |
| `RT` | Recorded response time |
| `bad_subject_zerohit` | Participant-level zero-hit exclusion flag |
| `trialID` | Trial number within participant-condition sequence |

The `hgf_inputs` subfolder contains the model-ready response sequences used
by the current Julia HGF fits.
