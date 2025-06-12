import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

async function runPythonPrediction(claimData: any) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(process.cwd(), 'models', 'single_model_openai_denial_predictor.py');
    const python = spawn('python3', [scriptPath, JSON.stringify(claimData), '--json']);
    let output = '';
    let error = '';

    python.stdout.on('data', (data) => { output += data.toString(); });
    python.stderr.on('data', (data) => { error += data.toString(); });

    python.on('close', (code) => {
      if (code !== 0) return reject(new Error(error));
      try {
        const result = JSON.parse(output.trim());
        resolve(result);
      } catch (e) {
        reject(new Error('Failed to parse Python output: ' + output));
      }
    });
  });
}

export async function POST(request: NextRequest) {
  try {
    const data = await request.json();
    const predictionJson: any = await runPythonPrediction(data);
    return NextResponse.json({
      success: true,
      prediction: predictionJson,
    });
  } catch (error) {
    console.error('Prediction error:', error);
    return NextResponse.json(
      { error: 'Error generating prediction', details: error?.toString() },
      { status: 500 }
    );
  }
} 