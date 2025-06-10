import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { claim } from '@/lib/db/schema';
import { eq } from 'drizzle-orm';
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
    console.log('Prediction input:', data);
    // Skip DB insert for now
    const predictionJson: any = await runPythonPrediction(data);
    console.log('Prediction output:', predictionJson);
    return NextResponse.json({
      success: true,
      data: { id: 1 }, // fake id for review page
      prediction: predictionJson,
    });
  } catch (error) {
    console.error('Error (bypassing DB):', error);
    return NextResponse.json(
      { error: 'Error generating prediction', details: error?.toString() },
      { status: 500 }
    );
  }
}

export async function PATCH(request: NextRequest) {
  try {
    const data = await request.json();
    const { id, prediction, confidence_score, likelihood_percent, denial_reasons, next_steps, analysis_details } = data;
    // Update prediction fields for the claim
    const updated = await db.update(claim)
      .set({
        prediction,
        confidenceScore: confidence_score,
        likelihoodPercent: likelihood_percent,
        denialReasons: denial_reasons,
        nextSteps: next_steps,
        analysisDetails: analysis_details,
      })
      .where(eq(claim.id, id))
      .returning();
    return NextResponse.json({ success: true, data: updated[0] });
  } catch (error) {
    console.error('Error updating claim prediction:', error);
    return NextResponse.json(
      { error: 'Error updating claim prediction' },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');
    if (!id) {
      return NextResponse.json({ error: 'Missing claim id' }, { status: 400 });
    }
    const result = await db.select().from(claim).where(eq(claim.id, Number(id)));
    if (!result[0]) {
      return NextResponse.json({ error: 'Claim not found' }, { status: 404 });
    }
    return NextResponse.json({ success: true, data: result[0] });
  } catch (error) {
    console.error('Error fetching claim:', error);
    return NextResponse.json(
      { error: 'Error fetching claim' },
      { status: 500 }
    );
  }
} 