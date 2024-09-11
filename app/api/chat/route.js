import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenerativeAI } from "@google/generative-ai";

const systemPrompt = `
You are an AI assistant designed to help evaluate and provide feedback on teacher performance based on student ratings and comments. Your role is to analyze the data provided, identify patterns and trends, and offer constructive insights to help teachers improve their teaching methods and student engagement.

You will be given some relevant professor reviews from our database. Use this information to inform your response, but also feel free to generalize and provide broader insights when appropriate.

Key Responsibilities:
- Analyze numerical ratings across various teaching aspects (e.g., clarity, engagement, fairness).
- Interpret qualitative feedback from student comments.
- Identify strengths and areas for improvement in teaching performance.
- Provide actionable recommendations for professional development.
- Maintain objectivity and fairness in your assessments.
- Respect student and teacher privacy by not identifying specific individuals beyond what's provided in the reviews.
`;

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;

if (!GOOGLE_API_KEY) {
  throw new Error("GOOGLE_API_KEY is not set in environment variables");
}
if (!PINECONE_API_KEY) {
  throw new Error("PINECONE_API_KEY is not set in environment variables");
}

const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY);
const pc = new Pinecone({ apiKey: PINECONE_API_KEY });

export async function POST(req) {
    try {
        const data = await req.json();
        const index = pc.Index('rag');
        const text = data[data.length-1].content;
        
        // Generate embedding for the query
        const embeddingModel = genAI.getGenerativeModel({ model: "embedding-001" });
        const embeddingResult = await embeddingModel.embedContent(text);
        const embedding = embeddingResult['embedding'];
        
        // Query Pinecone
        const queryResponse = await index.query({
            vector: embedding,
            topK: 3,
            includeMetadata: true
        });

        let contextString = 'Relevant professor reviews:\n\n';
        queryResponse.matches.forEach((match) => {
            contextString += `
            Professor: ${match.id}
            Review: ${match.metadata.review}
            Subject: ${match.metadata.subject}
            Stars: ${match.metadata.stars}
            \n\n`;
        });

        // Prepare the prompt for Gemini
        const promptForGemini = `${systemPrompt}\n\nContext:\n${contextString}\n\nUser Query: ${text}\n\nPlease provide an analysis and response based on the above context and query:`;

        // Use Gemini AI for text generation
        const model = genAI.getGenerativeModel({ model: "gemini-pro" });
        const result = await model.generateContent(promptForGemini);
        const generatedText = result.response.text();

        return new NextResponse(generatedText);
    } catch (error) {
        console.error("Error in POST request:", error);
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}