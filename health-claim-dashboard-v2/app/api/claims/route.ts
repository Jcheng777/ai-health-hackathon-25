import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { claim } from '@/lib/db/schema';
import { spawn } from 'child_process';
import path from 'path';

async function runPythonPrediction(claimData: any) {
  return new Promise((resolve, reject) => {
    // Adjust the path if your script is elsewhere
    const scriptPath = path.join(process.cwd(), 'models', 'openai_denial_predictor.py');
    const python = spawn('python3', [scriptPath, JSON.stringify(claimData)]);
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

    // 1. Store in DB
    const dbResult = await db.insert(claim).values({
      providerId: data.provider_id,
      procedureCode: data.procedure_code,
      diagnosisCode: data.diagnosis_code,
      billedAmount: data.billed_amount,
      insuranceType: data.insurance_type,
      additionalInfo: data.additional_info,
    }).returning();

    // 2. Run Python prediction
    const prediction = await runPythonPrediction(data);

    // 3. Return both DB and prediction result
    return NextResponse.json({
      success: true,
      data: dbResult[0],
      prediction, // { prediction: ..., confidence: ..., ... }
    });
  } catch (error) {
    console.error('Error storing claim or running prediction:', error);
    return NextResponse.json(
      { error: 'Error storing claim or running prediction' },
      { status: 500 }
    );
  }
} 