import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'
import { neon } from '@neondatabase/serverless'
import { drizzle } from 'drizzle-orm/neon-http'
import * as schema from './schema'
import { db } from '@/lib/db'
import { claims } from '@/lib/db/schema'

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

const sql = neon(process.env.DATABASE_URL!)
export const db = drizzle(sql, { schema })

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

    // Only use Python prediction model
    const prediction = await runPredictionModel(claimData)

    const result = await db.insert(claims).values({
      // ...fields
    }).returning()

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