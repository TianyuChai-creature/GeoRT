# PC1 beta1 anchor-direction fix

```json
{
  "inputs": {
    "new_bundle": "data/anchors_custom_right_arc_bending_v3_pc1beta1_lateral085_ringmono_frozenrobot.npz",
    "new_bundle_sha256": "3bf9a190cac67cd54a6dd270adc769da01fd621218fab19c51947640bb0ca6ba",
    "old_bundle": "data/anchors_custom_right_arc_bending_v2_lateral085_ringmono_frozenrobot.npz",
    "old_bundle_sha256": "f291cfc39c97bdda9e50bb670c7c14967428d3b9cbf7d83d9800fb43de51ed7e"
  },
  "new_h_over_r": [
    {
      "finger": "thumb",
      "human_over_robot": 0.938954178950789,
      "human_span_normalized": 1.3085184265391827,
      "robot_span_normalized": 1.3935913550130359,
      "type": "lateral"
    },
    {
      "finger": "thumb",
      "human_over_robot": 0.8099964653528837,
      "human_span_normalized": 1.6762472975200424,
      "robot_span_normalized": 2.069450138637046,
      "type": "bending"
    },
    {
      "finger": "index",
      "human_over_robot": 0.622268414192995,
      "human_span_normalized": 0.9649825611990581,
      "robot_span_normalized": 1.550749707343769,
      "type": "lateral"
    },
    {
      "finger": "index",
      "human_over_robot": 0.973731286088887,
      "human_span_normalized": 2.0817852477668555,
      "robot_span_normalized": 2.1379463487597334,
      "type": "bending"
    },
    {
      "finger": "middle",
      "human_over_robot": 0.614632916214914,
      "human_span_normalized": 0.5594908204438384,
      "robot_span_normalized": 0.9102845058955572,
      "type": "lateral"
    },
    {
      "finger": "middle",
      "human_over_robot": 0.9492197431926782,
      "human_span_normalized": 2.028464246859528,
      "robot_span_normalized": 2.1369806742923787,
      "type": "bending"
    },
    {
      "finger": "ring",
      "human_over_robot": 0.6632993384134398,
      "human_span_normalized": 0.6149433953118709,
      "robot_span_normalized": 0.9270978571797878,
      "type": "lateral"
    },
    {
      "finger": "ring",
      "human_over_robot": 0.9477752978264368,
      "human_span_normalized": 2.028331904518849,
      "robot_span_normalized": 2.1400978788648395,
      "type": "bending"
    },
    {
      "finger": "pinky",
      "human_over_robot": 0.8162264482427508,
      "human_span_normalized": 1.0654718115868906,
      "robot_span_normalized": 1.305363007876967,
      "type": "lateral"
    },
    {
      "finger": "pinky",
      "human_over_robot": 0.9448564191764873,
      "human_span_normalized": 2.0646639246905663,
      "robot_span_normalized": 2.1851615576576964,
      "type": "bending"
    }
  ],
  "new_h_over_r_scatter": {
    "max": 0.973731286088887,
    "max_over_min": 1.5842485171236906,
    "min": 0.614632916214914
  },
  "new_l1_l5_direction": [
    {
      "finger": "thumb",
      "human_beta1_rad": 0.2132840396013977,
      "human_beta2_rad": -0.1363154090233875,
      "human_beta3_rad": -0.015980031746084356,
      "level": 1,
      "robot_alpha_rad": 0.0,
      "robot_beta1_rad": -0.3499999940395355,
      "robot_beta2_rad": -0.3499999940395355,
      "robot_beta3_rad": -0.17499999701976776,
      "source_frame": 2682,
      "tip_arc_fraction": 0.020548339795802926
    },
    {
      "finger": "thumb",
      "human_beta1_rad": 0.7744495891389794,
      "human_beta2_rad": -0.2303398681538367,
      "human_beta3_rad": -0.2783053080539971,
      "level": 5,
      "robot_alpha_rad": 0.0,
      "robot_beta1_rad": 0.7850000262260437,
      "robot_beta2_rad": 0.7850000262260437,
      "robot_beta3_rad": 0.39250001311302185,
      "source_frame": 150010,
      "tip_arc_fraction": 0.838219826147389
    },
    {
      "finger": "index",
      "human_beta1_rad": -0.1744203998116417,
      "human_beta2_rad": 0.011784920668735755,
      "human_beta3_rad": -0.16557038119299652,
      "level": 1,
      "robot_alpha_rad": 0.0,
      "robot_beta1_rad": 0.0,
      "robot_beta2_rad": 0.0,
      "robot_beta3_rad": 0.0,
      "source_frame": 52592,
      "tip_arc_fraction": 0.0
    },
    {
      "finger": "index",
      "human_beta1_rad": 0.7651599704638544,
      "human_beta2_rad": 1.226191616992263,
      "human_beta3_rad": -0.5335628752254351,
      "level": 5,
      "robot_alpha_rad": 0.0,
      "robot_beta1_rad": 1.5700000524520874,
      "robot_beta2_rad": 1.5700000524520874,
      "robot_beta3_rad": 0.7850000262260437,
      "source_frame": 41168,
      "tip_arc_fraction": 1.0
    },
    {
      "finger": "middle",
      "human_beta1_rad": -0.11861969809291006,
      "human_beta2_rad": 0.034387275835180635,
      "human_beta3_rad": -0.19009593255410157,
      "level": 1,
      "robot_alpha_rad": 0.0,
      "robot_beta1_rad": 0.0,
      "robot_beta2_rad": 0.0,
      "robot_beta3_rad": 0.0,
      "source_frame": 56416,
      "tip_arc_fraction": 0.0
    },
    {
      "finger": "middle",
      "human_beta1_rad": 1.2212143541755496,
      "human_beta2_rad": 1.4062683101154252,
      "human_beta3_rad": -0.9186445538255438,
      "level": 5,
      "robot_alpha_rad": 0.0,
      "robot_beta1_rad": 1.5700000524520874,
      "robot_beta2_rad": 1.5700000524520874,
      "robot_beta3_rad": 0.7850000262260437,
      "source_frame": 134742,
      "tip_arc_fraction": 1.0
    },
    {
      "finger": "ring",
      "human_beta1_rad": -0.07403150900758046,
      "human_beta2_rad": 0.04483273384090194,
      "human_beta3_rad": -0.08136508562149442,
      "level": 1,
      "robot_alpha_rad": 0.0,
      "robot_beta1_rad": 0.0,
      "robot_beta2_rad": 0.0,
      "robot_beta3_rad": 0.0,
      "source_frame": 58972,
      "tip_arc_fraction": 0.0
    },
    {
      "finger": "ring",
      "human_beta1_rad": 1.358797309121998,
      "human_beta2_rad": 1.3957188247466576,
      "human_beta3_rad": -0.912108532094721,
      "level": 5,
      "robot_alpha_rad": 0.0,
      "robot_beta1_rad": 1.5700000524520874,
      "robot_beta2_rad": 1.5700000524520874,
      "robot_beta3_rad": 0.7850000262260437,
      "source_frame": 93278,
      "tip_arc_fraction": 1.0
    },
    {
      "finger": "pinky",
      "human_beta1_rad": -0.20998227162895045,
      "human_beta2_rad": 0.20032194475519385,
      "human_beta3_rad": -0.07692048323421188,
      "level": 1,
      "robot_alpha_rad": 0.0,
      "robot_beta1_rad": 0.0,
      "robot_beta2_rad": 0.0,
      "robot_beta3_rad": 0.0,
      "source_frame": 21810,
      "tip_arc_fraction": 0.0
    },
    {
      "finger": "pinky",
      "human_beta1_rad": 1.3052349988976948,
      "human_beta2_rad": 0.7198953419111307,
      "human_beta3_rad": -0.9968207655475402,
      "level": 5,
      "robot_alpha_rad": 0.0,
      "robot_beta1_rad": 1.5700000524520874,
      "robot_beta2_rad": 1.5700000524520874,
      "robot_beta3_rad": 0.7850000262260437,
      "source_frame": 40124,
      "tip_arc_fraction": 1.0
    }
  ],
  "protocol": {
    "h_over_r": "norm(N_human(L5)-N_human(L1)) / norm(N_robot(L5)-N_robot(L1))",
    "units": {
      "joint": "rad",
      "residual": "m"
    }
  },
  "residual_m": {
    "c2b_s42": {
      "new": [
        {
          "count": 150,
          "finger": "thumb",
          "max_m": 0.09619199484586716,
          "mean_m": 0.02945534698665142
        },
        {
          "count": 150,
          "finger": "index",
          "max_m": 0.09014716744422913,
          "mean_m": 0.03313957154750824
        },
        {
          "count": 150,
          "finger": "middle",
          "max_m": 0.07366862893104553,
          "mean_m": 0.023802176117897034
        },
        {
          "count": 150,
          "finger": "ring",
          "max_m": 0.07535302639007568,
          "mean_m": 0.024775760248303413
        },
        {
          "count": 150,
          "finger": "pinky",
          "max_m": 0.07491210103034973,
          "mean_m": 0.023978589102625847
        }
      ],
      "old": [
        {
          "count": 150,
          "finger": "thumb",
          "max_m": 0.09619199484586716,
          "mean_m": 0.02945534698665142
        },
        {
          "count": 150,
          "finger": "index",
          "max_m": 0.11953966319561005,
          "mean_m": 0.062412604689598083
        },
        {
          "count": 150,
          "finger": "middle",
          "max_m": 0.07366862893104553,
          "mean_m": 0.023802176117897034
        },
        {
          "count": 150,
          "finger": "ring",
          "max_m": 0.07535302639007568,
          "mean_m": 0.024775760248303413
        },
        {
          "count": 150,
          "finger": "pinky",
          "max_m": 0.07491210103034973,
          "mean_m": 0.023978589102625847
        }
      ]
    },
    "c2el_s42": {
      "new": [
        {
          "count": 150,
          "finger": "thumb",
          "max_m": 0.08307743817567825,
          "mean_m": 0.03070928528904915
        },
        {
          "count": 150,
          "finger": "index",
          "max_m": 0.088878333568573,
          "mean_m": 0.033291351050138474
        },
        {
          "count": 150,
          "finger": "middle",
          "max_m": 0.046291619539260864,
          "mean_m": 0.016758717596530914
        },
        {
          "count": 150,
          "finger": "ring",
          "max_m": 0.06736232340335846,
          "mean_m": 0.02651972696185112
        },
        {
          "count": 150,
          "finger": "pinky",
          "max_m": 0.07160086929798126,
          "mean_m": 0.02897043153643608
        }
      ],
      "old": [
        {
          "count": 150,
          "finger": "thumb",
          "max_m": 0.08307743817567825,
          "mean_m": 0.03070928528904915
        },
        {
          "count": 150,
          "finger": "index",
          "max_m": 0.11409566551446915,
          "mean_m": 0.060210954397916794
        },
        {
          "count": 150,
          "finger": "middle",
          "max_m": 0.046291619539260864,
          "mean_m": 0.016758717596530914
        },
        {
          "count": 150,
          "finger": "ring",
          "max_m": 0.06736232340335846,
          "mean_m": 0.02651972696185112
        },
        {
          "count": 150,
          "finger": "pinky",
          "max_m": 0.07160086929798126,
          "mean_m": 0.02897043153643608
        }
      ]
    }
  }
}
```
