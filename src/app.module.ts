import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { GraphQLModule } from '@nestjs/graphql';
import { AuthModule } from './auth/auth.module';


@Module({
    imports: [
        GraphQLModule.forRoot({
            autoSchemaFile: true,
        }),
        AuthModule
    ],
    controllers: [AppController],
    providers: [AppService],
})
export class AppModule { }
