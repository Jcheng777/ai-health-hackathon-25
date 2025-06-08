import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

interface ClaimData {
  procedureCode: string
  diagnosisCode: string
  insuranceType: string
  providerId: string
  billedAmount: number
  allowedAmount: number
  paidAmount: number
  reasonCode?: string
  followUpRequired?: string
}

interface PredictionResult {
  prediction: 'approved' | 'denied' | 'review'
  confidence: number
  reasoning: string[]
  riskFactors: string[]
}

export async function POST(request: NextRequest) {
  try {
    const claimData: ClaimData = await request.json()

    // Validate required fields
    const requiredFields = ['procedureCode', 'diagnosisCode', 'insuranceType', 'billedAmount']
    for (const field of requiredFields) {
      if (!claimData[field as keyof ClaimData]) {
        return NextResponse.json(
          { error: `Missing required field: ${field}` },
          { status: 400 }
        )
      }
    }

    // Try Python prediction script first, fallback to rule-based
    let prediction: PredictionResult
    try {
      prediction = await runPredictionModel(claimData)
    } catch (mlError) {
      console.warn('ML prediction failed, using rule-based fallback:', mlError)
      prediction = getRuleBasedPrediction(claimData)
    }

    return NextResponse.json(prediction)
  } catch (error) {
    console.error('Prediction error:', error)
    return NextResponse.json(
      { error: 'Failed to generate prediction' },
      { status: 500 }
    )
  }
}

function runPredictionModel(claimData: ClaimData): Promise<PredictionResult> {
  return new Promise((resolve, reject) => {
    // Path to the Python prediction script
    const scriptPath = path.join(process.cwd(), '..', 'scripts', 'predict_denial.py')
    
    // Spawn Python process
    const pythonProcess = spawn('python', [scriptPath, JSON.stringify(claimData)])
    
    let output = ''
    let errorOutput = ''

    pythonProcess.stdout.on('data', (data) => {
      output += data.toString()
    })

    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString()
    })

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python script failed: ${errorOutput}`))
        return
      }

      try {
        const result = JSON.parse(output.trim())
        resolve(result)
      } catch (parseError) {
        reject(new Error(`Failed to parse prediction result: ${parseError}`))
      }
    })

    pythonProcess.on('error', (error) => {
      reject(new Error(`Failed to start Python process: ${error.message}`))
    })
  })
}

// Fallback rule-based prediction if ML model is unavailable
function getRuleBasedPrediction(claimData: ClaimData): PredictionResult {
  const riskFactors: string[] = []
  let riskScore = 0

  // Rule 1: High amount claims are more likely to be reviewed
  if (claimData.billedAmount > 500) {
    riskFactors.push('High billed amount (>$500)')
    riskScore += 0.3
  }

  // Rule 2: Certain procedure codes have higher denial rates
  const highRiskProcedures = ['99238', '99233', '99232']
  if (highRiskProcedures.includes(claimData.procedureCode)) {
    riskFactors.push('High-risk procedure code')
    riskScore += 0.4
  }

  // Rule 3: Insurance type affects approval
  if (claimData.insuranceType === 'Self-Pay') {
    riskFactors.push('Self-pay insurance type')
    riskScore += 0.2
  }

  // Rule 4: Large discrepancy between billed and allowed
  if (claimData.allowedAmount && claimData.billedAmount) {
    const discrepancy = (claimData.billedAmount - claimData.allowedAmount) / claimData.billedAmount
    if (discrepancy > 0.3) {
      riskFactors.push('Large discrepancy between billed and allowed amounts')
      riskScore += 0.3
    }
  }

  // Determine prediction based on risk score
  let prediction: 'approved' | 'denied' | 'review'
  let reasoning: string[]

  if (riskScore >= 0.7) {
    prediction = 'denied'
    reasoning = ['High risk score indicates likely denial', ...riskFactors]
  } else if (riskScore >= 0.4) {
    prediction = 'review'
    reasoning = ['Medium risk score suggests manual review needed', ...riskFactors]
  } else {
    prediction = 'approved'
    reasoning = ['Low risk score indicates likely approval']
  }

  return {
    prediction,
    confidence: Math.min(0.85, 0.5 + riskScore), // Cap confidence for rule-based
    reasoning,
    riskFactors
  }
} 