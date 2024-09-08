import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import Groq from "groq";

const systemPrompt = 
`
You are an AI assistant designed to help evaluate and provide feedback on teacher performance based on student ratings and comments. Your role is to analyze the data provided, identify patterns and trends, and offer constructive insights to help teachers improve their teaching methods and student engagement.
Key Responsibilities:

Analyze numerical ratings across various teaching aspects (e.g., clarity, engagement, fairness).
Interpret qualitative feedback from student comments.
Identify strengths and areas for improvement in teaching performance.
Provide actionable recommendations for professional development.
Maintain objectivity and fairness in your assessments.
Respect student and teacher privacy by not identifying individuals.
`
export async function POST(req) {
    const data = await req.json()
    const pc=new Pinecone({
        apiKey: '83dd7137-dace-465a-a81e-70aa4358ab3a',
    })    
    const index = pc.index('rag').name('ns1')
    const Groq = new Groq()

    const text = data[data.length-1].content
    const embedding = await Groq.Embeddings.create({
        model: 'text-embedding-ada-002',
        input:text,
        encoding_format: 'float',
    })

    const results = await index.query({
        topK:3,
        includeMetadata: true,
        vector: embedding.data[0].embedding
    })

    let resultString = 'Returned results:'
    results.matches.forEach((match)=>{
        resultString += `
        Professor:${match.id}
        Review: ${match.metadata.stars}
        Subject:${match.metadata.subject}
        Stars:${match.metadata.stars}
        \n\n
        `
    })
    const lastMessage = data[data.length-1]
    const lastMessageContent = lastMessage.content + resultsString
    const lastDataWithoutLastMessage = data.slice(0,data.length-1)
    const completion = await groq.chat.completions.create({
        messages: [
            {role:'system',content:systemPrompt},
            ...lastDataWithoutLastMessage,
            {role:'user',content:lastMessageContent}
        ],
        model:'llama3-8b-8192',
        stream:true,
    })
    const stream =ReadableStream({
        async start(controller){
            const encoder= new TextEncoder()
            try{
                for await(const chunk of completion ){
                    const content=chunk.choices[0]?.delta?.content
                    if(content){
                        const text=encoder.encode(content)
                        controller.enqueue(text)
                    }
                }
            }
            catch(err){
                controller.err(err)
            }
            finally{
                controller.close()
            }
        }
    })
    return new NextResponse(stream)

}