#![allow(dead_code)]

use neuralnet::data_handling;
use neuralnet::layers::Layer1D;
use neuralnet::activation_fn::Activation;

fn main() {
    // Read CSV of rows of strings -> Vec<Vec<String>>
    let data = data_handling::read_csv("data.csv").expect("failed to read data.csv");

    // Example expects rows with at least 3 columns: x0, x1, y
    let mut xdash: Vec<[f64; 3]> = Vec::new(); // 2 features + bias
    let mut ydash: Vec<f64> = Vec::new();

    for row in data.iter() {
        if row.len() < 3 { continue; }
        let x0 = row[0].parse::<f64>().unwrap_or(0.0);
        let x1 = row[1].parse::<f64>().unwrap_or(0.0);
        let y = row[2].parse::<f64>().unwrap_or(0.0);
        xdash.push([x0, x1, 1.0]); // last entry is bias input = 1.0
        ydash.push(y);
    }

    // Example shapes: hidden layer = 3 hidden units, input size = 3 (including bias)
    // output layer produces 1 scalar (OUT=1) from 3 hidden units (IN=3)
    let mut hidden_layer = Layer1D::<f64, 3, 3>::new(
        [[0.5; 3]; 3], // weights for 3 outputs each with 3 inputs (for example)
        [0.1; 3],
    );

    // output layer: OUT = 1, IN = 3
    let mut output_layer = Layer1D::<f64, 1, 3>::new([[0.5, 0.5, 0.5]], [0.1]);

    let activation = Activation::Sigmoid;
    let lr = 0.01f64;

    let epochs = 100_000usize;
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f64;

        for (input, &target) in xdash.iter().zip(ydash.iter()) {
            // forward
            let hidden_out = hidden_layer.forward(input);         // [f64; 3]
            let hidden_act = activation.forward(&hidden_out);     // [f64; 3]
            let final_out = output_layer.forward(&hidden_act);    // [f64; 1]
            let pred_pre = final_out[0];
            let pred = activation.forward(&final_out)[0];

            // compute scalar loss (MSE) and accumulate
            let error = pred - target;
            epoch_loss += error * error;

            // derivative w.r.t. pre-activation of output: dL/dz = dL/dy * dy/dz
            // dL/dy = 2 * (y - t)  (we'll incorporate scalar factor into learning rate or keep as-is)
            let d_loss_dy = 2.0 * (pred - target);
            let dy_dz = activation.derivative(pred_pre);
            let d_out = d_loss_dy * dy_dz;

            // weight grads for output layer (shape [1][3])
            let mut output_weight_grads = [[0.0f64; 3]; 1];
            for j in 0..3 {
                output_weight_grads[0][j] = d_out * hidden_act[j];
            }
            let output_bias_grads = [d_out];

            // update output layer
            output_layer.update_weights(&output_weight_grads, &output_bias_grads, lr);

            // backprop into hidden layer: compute d_hidden for each hidden neuron
            let mut hidden_weight_grads = [[0.0f64; 3]; 3]; // shape [OUT_hidden=3][IN=3]
            let mut hidden_bias_grads = [0.0f64; 3];
            for h in 0..3 {
                // weight from hidden h to output 0 is output_layer.weights[0][h]
                let w_ho = output_layer.weights[0][h];
                let d_hidden = d_out * w_ho * activation.derivative(hidden_out[h]);
                // gradient for each input weight to hidden neuron h: d_hidden * input[k]
                for k in 0..3 {
                    hidden_weight_grads[h][k] = d_hidden * input[k];
                }
                hidden_bias_grads[h] = d_hidden;
            }
            hidden_layer.update_weights(&hidden_weight_grads, &hidden_bias_grads, lr);
        }

        // display training outputs every 1000 epochs
        if epoch % 1000 == 0 {
            let n = (ydash.len().max(1)) as f64;
            let mean_loss = epoch_loss / n;
            println!("Epoch {}/{} - Loss = {}", epoch, epochs, mean_loss);
        }
    }
}
