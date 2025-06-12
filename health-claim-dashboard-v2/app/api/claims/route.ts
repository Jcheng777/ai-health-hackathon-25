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
    // Save all claim and prediction data provided in the request
    const [newClaim] = await db.insert(claim)
      .values({
        providerId: data.provider_id,
        procedureCode: data.procedure_code,
        diagnosisCode: data.diagnosis_code,
        billedAmount: data.billed_amount,
        insuranceType: data.insurance_type,
        additionalInfo: data.additional_info,
        prediction: data.prediction,
        confidenceScore: data.confidence_percent,
        likelihoodPercent: data.likelihood_percent,
        denialReasons: data.denial_reasons,
        acceptedReasons: data.accepted_reasons,
        nextSteps: data.next_steps,
        analysisDetails: data.analysis_details,
      })
      .returning();

    return NextResponse.json({
      success: true,
      data: newClaim,
    });
  } catch (error) {
    console.error('Error:', error);
    return NextResponse.json(
      { error: 'Error saving claim', details: error?.toString() },
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
    if (id) {
      const result = await db.select().from(claim).where(eq(claim.id, Number(id)));
      if (!result[0]) {
        return NextResponse.json({ error: 'Claim not found' }, { status: 404 });
      }
      return NextResponse.json({ success: true, data: result[0] });
    } else {
      // Fetch all claims
      const results = await db.select().from(claim);
      return NextResponse.json({ success: true, data: results });
    }
  } catch (error) {
    console.error('Error fetching claim(s):', error);
    return NextResponse.json(
      { error: 'Error fetching claim(s)' },
      { status: 500 }
    );
  }
} 